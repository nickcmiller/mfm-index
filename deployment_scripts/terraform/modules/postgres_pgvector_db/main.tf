resource "google_sql_database_instance" "postgres_pgvector_instance" {
  name                = var.INSTANCE_NAME
  database_version    = var.DATABASE_VERSION
  region              = var.REGION
  deletion_protection = var.DELETION_PROTECTION

  settings {
    tier = var.TIER
  }
}

resource "google_sql_user" "admin" {
  instance  = google_sql_database_instance.postgres_pgvector_instance.name
  name      = var.ADMIN_USER
  password  = var.ADMIN_PASSWORD
  depends_on = [google_sql_database_instance.postgres_pgvector_instance]
}

resource "google_sql_database" "postgres_pgvector_db" {
  name       = var.DATABASE_NAME
  instance   = google_sql_database_instance.postgres_pgvector_instance.name
  depends_on = [google_sql_database_instance.postgres_pgvector_instance]
}

variable "INSTANCE_NAME" {}
variable "DATABASE_VERSION" {}
variable "REGION" {}
variable "DELETION_PROTECTION" {}
variable "TIER" {}
variable "ADMIN_USER" {}
variable "ADMIN_PASSWORD" {}
variable "DATABASE_NAME" {}

output "instance_name" {
  value = google_sql_database_instance.postgres_pgvector_instance.name
}