#!/bin/bash

# Azure Container Apps Deployment Script for Veena TTS
# This script automates the deployment of the TTS application to Azure Container Apps

set -e

# Configuration variables
RESOURCE_GROUP="veena-tts-rg"
LOCATION="eastus2"
ACR_NAME="veenattscr"
STORAGE_ACCOUNT="veenattsstorage"
KEY_VAULT_NAME="veena-tts-kv"
CONTAINER_APP_ENV="veena-tts-env"
CONTAINER_APP_NAME="veena-tts-app"
IMAGE_NAME="veena-tts"
IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_prerequisites() {
    echo_info "Checking prerequisites..."
    
    if ! command -v az &> /dev/null; then
        echo_error "Azure CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        echo_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if logged into Azure
    if ! az account show &> /dev/null; then
        echo_error "Not logged into Azure. Please run 'az login' first."
        exit 1
    fi
    
    echo_info "Prerequisites check passed!"
}

# Create Azure resources
create_azure_resources() {
    echo_info "Creating Azure resources..."
    
    # Create resource group
    echo_info "Creating resource group: $RESOURCE_GROUP"
    az group create --name $RESOURCE_GROUP --location $LOCATION
    
    # Create Azure Container Registry
    echo_info "Creating Azure Container Registry: $ACR_NAME"
    az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Premium --location $LOCATION
    
    # Create Key Vault
    echo_info "Creating Key Vault: $KEY_VAULT_NAME"
    az keyvault create --name $KEY_VAULT_NAME --resource-group $RESOURCE_GROUP --location $LOCATION --enable-rbac-authorization
    
    # Create Storage Account for model cache
    echo_info "Creating Storage Account: $STORAGE_ACCOUNT"
    az storage account create \
        --name $STORAGE_ACCOUNT \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --sku Standard_LRS \
        --kind StorageV2
    
    # Create file share for model cache
    echo_info "Creating file share for model cache"
    az storage share create \
        --name model-cache \
        --account-name $STORAGE_ACCOUNT \
        --quota 1000
    
    echo_info "Azure resources created successfully!"
}

# Set up secrets
setup_secrets() {
    echo_info "Setting up secrets..."
    
    if [ -z "$HF_TOKEN" ]; then
        echo_warn "HF_TOKEN environment variable not set. Please enter your HuggingFace token:"
        read -s HF_TOKEN
    fi
    
    if [ -z "$HF_TOKEN" ]; then
        echo_error "HuggingFace token is required!"
        exit 1
    fi
    
    # Get current user's principal ID for Key Vault access
    echo_info "Configuring Key Vault access for current user..."
    CURRENT_USER_ID=$(az ad signed-in-user show --query id -o tsv)
    
    # Grant Key Vault Secrets Officer role to current user
    echo_info "Granting Key Vault Secrets Officer role to current user"
    az role assignment create \
        --assignee $CURRENT_USER_ID \
        --role "Key Vault Secrets Officer" \
        --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$KEY_VAULT_NAME" || echo_warn "Role assignment might already exist"
    
    # Wait for role assignment to propagate
    echo_info "Waiting for role assignment to propagate (30 seconds)..."
    sleep 30
    
    # Add HuggingFace token to Key Vault
    echo_info "Adding HuggingFace token to Key Vault"
    az keyvault secret set \
        --vault-name $KEY_VAULT_NAME \
        --name "hf-token" \
        --value "$HF_TOKEN"
    
    echo_info "Secrets configured successfully!"
}

# Build and push Docker image
build_and_push_image() {
    echo_info "Building and pushing Docker image..."
    
    # Login to ACR
    echo_info "Logging into Azure Container Registry"
    az acr login --name $ACR_NAME
    
    # Check if we're on Apple Silicon and need cross-platform build
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        echo_info "Detected Apple Silicon, using cross-platform build for AMD64"
        
        # Create builder if it doesn't exist
        docker buildx create --use --name multiarch 2>/dev/null || docker buildx use multiarch
        
        # Build for AMD64 platform and push
        echo_info "Building Docker image for linux/amd64: $IMAGE_NAME:$IMAGE_TAG"
        docker buildx build --platform linux/amd64 \
            -t $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG \
            --push .
    else
        # Regular build for x86_64 systems
        echo_info "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
        docker build -t $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG .
        
        # Push Docker image
        echo_info "Pushing Docker image to registry"
        docker push $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG
    fi
    
    echo_info "Docker image built and pushed successfully!"
}

# Create Container App Environment
create_container_app_environment() {
    echo_info "Creating Container App Environment with GPU support..."
    
    # Check if GPU workload profiles are available in the region
    az containerapp env create \
        --name $CONTAINER_APP_ENV \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --enable-workload-profiles
    
    echo_info "Container App Environment created successfully!"
}

# Deploy Container App
deploy_container_app() {
    echo_info "Deploying Container App..."
    
    # Get storage account key
    STORAGE_KEY=$(az storage account keys list \
        --resource-group $RESOURCE_GROUP \
        --account-name $STORAGE_ACCOUNT \
        --query '[0].value' -o tsv)
    
    # Create the container app with GPU profile
    az containerapp create \
        --name $CONTAINER_APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --environment $CONTAINER_APP_ENV \
        --image $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG \
        --target-port 8000 \
        --ingress external \
        --workload-profile-name "GPU" \
        --cpu 4 \
        --memory 32Gi \
        --min-replicas 0 \
        --max-replicas 3 \
        --env-vars \
            "HF_TOKEN=secretref:hf-token" \
            "AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT" \
            "AZURE_STORAGE_SHARE=model-cache" \
            "AZURE_KEY_VAULT_URL=https://$KEY_VAULT_NAME.vault.azure.net/" \
            "PYTORCH_CUDA_ERROR_REPORTING=0" \
            "VLLM_ENGINE_ITERATION_TIMEOUT_S=120" \
            "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
        --secrets \
            "hf-token=keyvaultref:https://$KEY_VAULT_NAME.vault.azure.net/secrets/hf-token,identityref:system" \
        --registry-server $ACR_NAME.azurecr.io \
        --registry-identity system
    
    # Configure file share mount
    echo_info "Configuring Azure File Share mount..."
    az containerapp update \
        --name $CONTAINER_APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --set-env-vars \
            "AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=$STORAGE_ACCOUNT;AccountKey=$STORAGE_KEY;EndpointSuffix=core.windows.net"
    
    echo_info "Container App deployed successfully!"
}

# Configure managed identity and permissions
configure_identity() {
    echo_info "Configuring managed identity and permissions..."
    
    # Enable system-assigned managed identity
    az containerapp identity assign \
        --name $CONTAINER_APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --system-assigned
    
    # Get the managed identity principal ID
    PRINCIPAL_ID=$(az containerapp identity show \
        --name $CONTAINER_APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --query principalId -o tsv)
    
    # Grant Key Vault permissions using RBAC
    echo_info "Granting Key Vault permissions to managed identity"
    az role assignment create \
        --assignee $PRINCIPAL_ID \
        --role "Key Vault Secrets User" \
        --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$KEY_VAULT_NAME"
    
    # Grant ACR pull permissions
    echo_info "Granting ACR pull permissions to managed identity"
    az role assignment create \
        --assignee $PRINCIPAL_ID \
        --role "AcrPull" \
        --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ContainerRegistry/registries/$ACR_NAME"
    
    echo_info "Identity and permissions configured successfully!"
}

# Get deployment information
get_deployment_info() {
    echo_info "Getting deployment information..."
    
    FQDN=$(az containerapp show \
        --name $CONTAINER_APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --query properties.configuration.ingress.fqdn -o tsv)
    
    echo_info "🎉 Deployment completed successfully!"
    echo ""
    echo "=============================="
    echo "DEPLOYMENT INFORMATION"
    echo "=============================="
    echo "Application URL: https://$FQDN"
    echo "Status endpoint: https://$FQDN/status"
    echo "Speakers endpoint: https://$FQDN/speakers"
    echo "Health check: https://$FQDN/health"
    echo ""
    echo "Test the deployment:"
    echo "curl https://$FQDN/status"
    echo ""
    echo "Example TTS request:"
    echo "curl -N -X POST \"https://$FQDN/generate\" \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '{\"text\": \"नमस्ते, आपका दिन कैसा रहा?\", \"speaker_id\": \"vinaya_assist\", \"streaming\": true, \"output_format\": \"wav\", \"max_new_tokens\": 700}' \\"
    echo "  --output veena_output.wav"
    echo ""
    echo "Resource Group: $RESOURCE_GROUP"
    echo "Container App: $CONTAINER_APP_NAME"
    echo "Container Registry: $ACR_NAME"
    echo "=============================="
}

# Main deployment function
main() {
    echo_info "Starting Veena TTS deployment to Azure Container Apps..."
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-resources)
                SKIP_RESOURCES=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --resource-group)
                RESOURCE_GROUP="$2"
                shift
                shift
                ;;
            --location)
                LOCATION="$2"
                shift
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-resources    Skip Azure resource creation"
                echo "  --skip-build        Skip Docker image build"
                echo "  --resource-group    Specify resource group name"
                echo "  --location          Specify Azure location"
                echo "  -h, --help          Show this help message"
                exit 0
                ;;
            *)
                echo_error "Unknown option $1"
                exit 1
                ;;
        esac
    done
    
    check_prerequisites
    
    if [ "$SKIP_RESOURCES" != true ]; then
        create_azure_resources
        setup_secrets
    fi
    
    if [ "$SKIP_BUILD" != true ]; then
        build_and_push_image
    fi
    
    create_container_app_environment
    deploy_container_app
    configure_identity
    get_deployment_info
}

# Run main function
main "$@"
