variable "SQL_INSTANCE" {
  type = string
}

variable "DATABASE_VERSION" {
  type = string
}

variable "DEFAULT_REGION" {
  type = string
}

variable "DELETION_PROTECTION" {
  type = bool
}

variable "TIER" {
  type = string
}

variable "ADMIN_USER" {
  type = string
}

variable "ADMIN_PASSWORD" {
  type      = string
  sensitive = true
}

variable "DEFAULT_PROJECT" {
  type = string
}

variable "DEFAULT_ZONE" {
  type = string
}

variable "DATABASE_NAME" {
  type = string
}