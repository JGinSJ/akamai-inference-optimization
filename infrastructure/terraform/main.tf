terraform {
  required_version = ">= 1.6.0"

  required_providers {
    linode = {
      source  = "linode/linode"
      version = ">= 2.0.0, < 3.0.0"
    }
  }

  # TODO(open-question-1): Add a remote backend once CI/CD is in place.
  # Akamai Object Storage (S3-compatible) example:
  #
  # backend "s3" {
  #   bucket                      = "PLACEHOLDER-terraform-state"
  #   key                         = "lke/terraform.tfstate"
  #   region                      = "us-ord-1"
  #   endpoint                    = "PLACEHOLDER.linodeobjects.com"
  #   skip_credentials_validation = true
  #   skip_metadata_api_check     = true
  #   skip_region_validation      = true
  #   force_path_style            = true
  # }
}

provider "linode" {
  # Authenticate via the LINODE_TOKEN environment variable.
  # Never set a token value here or in any committed file.
  #
  # export LINODE_TOKEN="<your-personal-access-token>"
}
