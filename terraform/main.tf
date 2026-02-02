module "civicsight_container" {
  source = "./modules/docker_container"

  container_name = var.container_name
  image_name     = var.image_name
  internal_port  = var.internal_port
  external_port  = var.external_port
}