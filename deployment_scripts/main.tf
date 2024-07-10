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

resource "google_sql_database_instance" "mfm_index_sql_instance" {
  name             = "mfm-index-sql-instance"
  database_version = "POSTGRES_14"
  region           = var.DEFAULT_REGION
  deletion_protection = false

  settings {
    tier = "db-f1-micro"
  }
}

resource "google_sql_user" "admin" {
  instance = google_sql_database_instance.mfm_index_sql_instance.name
  name     = "admin"
  password = var.ADMIN_PASSWORD
  depends_on = [google_sql_database_instance.mfm_index_sql_instance]
}

resource "google_sql_database" "mfm_index_sql_database" {
  name = "mfm-index-sql-database"
  instance = google_sql_database_instance.mfm_index_sql_instance.name
  depends_on = [google_sql_database_instance.mfm_index_sql_instance]
}