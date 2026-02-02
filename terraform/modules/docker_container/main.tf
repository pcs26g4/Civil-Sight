resource "docker_image" "app_image" {
  name         = var.image_name
  keep_locally = true
}

resource "docker_container" "civicsight_container" {
  name  = var.container_name
  image = docker_image.app_image.image_id

  ports {
    internal = var.internal_port
    external = var.external_port
  }
}