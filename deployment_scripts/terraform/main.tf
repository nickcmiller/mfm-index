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

module "postgres_pgvector_db" {
  source = "./modules/postgres_pgvector_db"

  INSTANCE_NAME       = "mfm-index-sql-instance"
  DATABASE_VERSION    = "POSTGRES_14"
  REGION              = var.DEFAULT_REGION
  DELETION_PROTECTION = false
  TIER                = "db-f1-micro"
  ADMIN_USER          = "admin"
  ADMIN_PASSWORD      = var.ADMIN_PASSWORD
  DATABASE_NAME       = "mfm-index-sql-database"
}