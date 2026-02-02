variable "container_name" {
  default = "civicsight-local"
}

variable "image_name" {
  default = "nginx:latest"
}

variable "internal_port" {
  default = 80
}

variable "external_port" {
  default = 8026
}