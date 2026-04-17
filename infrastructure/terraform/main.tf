terraform {
  required_version = ">= 1.6.0"

  required_providers {
    linode = {
      source  = "linode/linode"
      version = ">= 2.0.0, < 3.0.0"
    }
  }

  # Remote state backend — Akamai Object Storage (S3-compatible).
  # Credentials are passed via environment variables at terraform init time:
  #   export AWS_ACCESS_KEY_ID="<object-storage-access-key>"
  #   export AWS_SECRET_ACCESS_KEY="<object-storage-secret-key>"
  # Never hardcode credentials here or in any committed file.
  backend "s3" {
    bucket = "inference-optimization"
    key    = "terraform.tfstate"
    region = "us-ord"

    endpoints = {
      s3 = "https://us-ord-1.linodeobjects.com"
    }

    skip_credentials_validation = true
    skip_metadata_api_check     = true
    skip_region_validation      = true
    skip_requesting_account_id  = true
    use_path_style              = true
  }
}

provider "linode" {
  # Authenticate via the LINODE_TOKEN environment variable.
  # Never set a token value here or in any committed file.
  #
  # export LINODE_TOKEN="<your-personal-access-token>"
}
