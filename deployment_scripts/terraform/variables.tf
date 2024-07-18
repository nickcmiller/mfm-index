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

variable "SQL_DATABASE" {
  description = "Same as DATABASE_NAME"
  type        = string
}

variable "GROQ_API_KEY" {
  description = "API key for GROQ"
  type        = string
}

variable "OPENAI_API_KEY" {
  description = "API key for OpenAI"
  type        = string
}

variable "SQL_HOST" {
  description = "SQL host"
  type        = string
}

variable "TABLE_NAME" {
  description = "Table name for vector storage"
  type        = string
}