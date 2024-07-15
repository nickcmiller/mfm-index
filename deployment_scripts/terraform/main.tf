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

  INSTANCE_NAME       = var.SQL_INSTANCE
  DATABASE_VERSION    = var.DATABASE_VERSION
  REGION              = var.DEFAULT_REGION
  DELETION_PROTECTION = var.DELETION_PROTECTION
  TIER                = var.TIER
  ADMIN_USER          = var.ADMIN_USER
  ADMIN_PASSWORD      = var.ADMIN_PASSWORD
  DATABASE_NAME       = var.DATABASE_NAME
}