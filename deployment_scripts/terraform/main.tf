terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.51.0"
    }
  }

  backend "gcs" {
    bucket  = "tf-for-mfm-index"
    prefix  = "terraform/state"
  }
}

provider "google" {
  project = var.DEFAULT_PROJECT
  region  = var.DEFAULT_REGION
  zone    = var.DEFAULT_ZONE
}

data "google_project" "project" {}

module "postgres_pgvector_db" {
  source = "./modules/postgres_pgvector_db"

  INSTANCE_NAME       = var.SQL_INSTANCE
  DATABASE_VERSION    = var.DATABASE_VERSION
  REGION              = var.DEFAULT_REGION
  DELETION_PROTECTION = var.DELETION_PROTECTION
  TIER                = var.TIER
  ADMIN_USER          = var.ADMIN_USER
  ADMIN_PASSWORD      = var.ADMIN_PASSWORD
  DATABASE_NAME       = var.DATABASE_NAME
}

resource "google_compute_router" "router" {
  name    = "cloud-router"
  network = "default"
  region  = var.DEFAULT_REGION
}

resource "google_compute_router_nat" "nat" {
  name                               = "cloud-nat"
  router                             = google_compute_router.router.name
  region                             = google_compute_router.router.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

resource "google_project_service" "vpc_access_api" {
  service = "vpcaccess.googleapis.com"
  project = var.DEFAULT_PROJECT

  disable_on_destroy = false
}

resource "google_vpc_access_connector" "connector" {
  name          = "vpc-connector"
  ip_cidr_range = "10.8.0.0/28"
  network       = "default"
  region        = var.DEFAULT_REGION

  depends_on = [google_project_service.vpc_access_api]
}

resource "google_service_account" "streamlit_sa" {
  account_id   = "streamlit-sa"
  display_name = "Streamlit Service Account"
}

resource "google_project_service" "run_api" {
  service = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "containerregistry_api" {
  service = "containerregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "sqladmin_api" {
  service = "sqladmin.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_iam_member" "streamlit_sa_cloudsql_client" {
  project = var.DEFAULT_PROJECT
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.streamlit_sa.email}"
}

resource "google_cloud_run_service" "backend_api" {
  name     = "backend-api"
  location = var.DEFAULT_REGION

  metadata {
    annotations = {
      "run.googleapis.com/ingress" = "internal"
    }
  }

  template {
    metadata {
      annotations = {
        "run.googleapis.com/vpc-access-connector" = google_vpc_access_connector.connector.name
        "run.googleapis.com/vpc-access-egress"    = "private-ranges-only"
      }
    }
    spec {
      service_account_name = google_service_account.streamlit_sa.email
      containers {
        image = "gcr.io/${var.DEFAULT_PROJECT}/backend-api"
        ports {
          container_port = 8000
        }
        env {
          name  = "GROQ_API_KEY"
          value = var.GROQ_API_KEY
        }
        env {
          name  = "OPENAI_API_KEY"
          value = var.OPENAI_API_KEY
        }
        env {
          name  = "SQL_DATABASE"
          value = var.SQL_DATABASE
        }
        env {
          name  = "SQL_HOST"
          value = var.SQL_HOST
        }
        env {
          name  = "SQL_INSTANCE"
          value = var.SQL_INSTANCE
        }

        env {
          name  = "ADMIN_USER"
          value = var.ADMIN_USER
        }
        env {
          name  = "ADMIN_PASSWORD"
          value = var.ADMIN_PASSWORD
        }
        env {
          name  = "TABLE_NAME"
          value = var.TABLE_NAME
        }
        env {
          name  = "INSTANCE_CONNECTION_NAME"
          value = "${var.DEFAULT_PROJECT}:${var.DEFAULT_REGION}:${var.SQL_INSTANCE}"
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  autogenerate_revision_name = true

  depends_on = [
    google_project_service.run_api, 
    google_service_account.streamlit_sa, 
    google_vpc_access_connector.connector, 
    google_project_service.sqladmin_api,
    google_project_iam_member.streamlit_sa_cloudsql_client
  ]
}

# Streamlit frontend service
resource "google_cloud_run_service" "streamlit_app" {
  name     = "streamlit-app"
  location = var.DEFAULT_REGION

  template {
    spec {
      service_account_name = google_service_account.streamlit_sa.email
      containers {
        image = "gcr.io/${var.DEFAULT_PROJECT}/streamlit-app"
        ports {
          container_port = 8501
        }
        env {
          name  = "BACKEND_URL"
          value = google_cloud_run_service.backend_api.status[0].url
        }
        env {
          name  = "TABLE_NAME"
          value = var.TABLE_NAME
        }
      }
    }
    metadata {
      annotations = {
        "run.googleapis.com/vpc-access-connector" = google_vpc_access_connector.connector.name
        "run.googleapis.com/vpc-access-egress"    = "all-traffic"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [google_project_service.run_api, google_cloud_run_service.backend_api, google_vpc_access_connector.connector]
}

# IAM policy to make the Streamlit app public
resource "google_cloud_run_service_iam_member" "streamlit_app_public" {
  service  = google_cloud_run_service.streamlit_app.name
  location = google_cloud_run_service.streamlit_app.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_service_iam_member" "backend_api_invoker_streamlit_sa" {
  service  = google_cloud_run_service.backend_api.name
  location = google_cloud_run_service.backend_api.location
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.streamlit_sa.email}"

  depends_on = [
    google_cloud_run_service.backend_api,
    google_service_account.streamlit_sa
  ]
}