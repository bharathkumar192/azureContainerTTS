#!/bin/bash

# Fix Key Vault permissions script
# This script fixes the RBAC permission issue for the Key Vault

set -e

# Configuration variables (should match your deployment)
RESOURCE_GROUP="veena-tts-rg"
KEY_VAULT_NAME="veena-tts-kv"

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

# Fix Key Vault permissions and add HuggingFace token
fix_keyvault_permissions() {
    echo_info "Fixing Key Vault permissions..."
    
    # Get HuggingFace token
    if [ -z "$HF_TOKEN" ]; then
        echo_warn "HF_TOKEN environment variable not set. Please enter your HuggingFace token:"
        read -s HF_TOKEN
    fi
    
    if [ -z "$HF_TOKEN" ]; then
        echo_error "HuggingFace token is required!"
        exit 1
    fi
    
    # Get current user's principal ID
    echo_info "Getting current user principal ID..."
    CURRENT_USER_ID=$(az ad signed-in-user show --query id -o tsv)
    echo_info "Current user ID: $CURRENT_USER_ID"
    
    # Grant Key Vault Secrets Officer role to current user
    echo_info "Granting Key Vault Secrets Officer role to current user..."
    az role assignment create \
        --assignee $CURRENT_USER_ID \
        --role "Key Vault Secrets Officer" \
        --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$KEY_VAULT_NAME" \
        2>/dev/null || echo_warn "Role assignment might already exist (this is normal)"
    
    # Wait for role assignment to propagate
    echo_info "Waiting for role assignment to propagate (30 seconds)..."
    sleep 30
    
    # Try to add HuggingFace token to Key Vault
    echo_info "Adding HuggingFace token to Key Vault..."
    retry_count=0
    max_retries=3
    
    while [ $retry_count -lt $max_retries ]; do
        if az keyvault secret set \
            --vault-name $KEY_VAULT_NAME \
            --name "hf-token" \
            --value "$HF_TOKEN" >/dev/null 2>&1; then
            echo_info "✅ HuggingFace token added successfully!"
            break
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo_warn "Failed to add secret, waiting 15 seconds before retry $retry_count/$max_retries..."
                sleep 15
            else
                echo_error "Failed to add secret after $max_retries attempts. Role propagation might take longer."
                echo_error "Please try running this script again in a few minutes."
                exit 1
            fi
        fi
    done
    
    echo_info "✅ Key Vault permissions fixed successfully!"
}

# Verify the fix worked
verify_keyvault_access() {
    echo_info "Verifying Key Vault access..."
    
    if az keyvault secret show --vault-name $KEY_VAULT_NAME --name "hf-token" >/dev/null 2>&1; then
        echo_info "✅ Successfully verified Key Vault access!"
        echo_info "You can now continue with the deployment by running:"
        echo ""
        echo "    ./azure_deploy.sh --skip-resources"
        echo ""
    else
        echo_error "❌ Key Vault access verification failed."
        echo_error "Please wait a few more minutes for role propagation and try again."
    fi
}

# Main function
main() {
    echo_info "🔧 Fixing Key Vault permissions for Veena TTS deployment..."
    echo ""
    
    # Check if logged into Azure
    if ! az account show &> /dev/null; then
        echo_error "Not logged into Azure. Please run 'az login' first."
        exit 1
    fi
    
    # Check if Key Vault exists
    if ! az keyvault show --name $KEY_VAULT_NAME --resource-group $RESOURCE_GROUP >/dev/null 2>&1; then
        echo_error "Key Vault $KEY_VAULT_NAME not found in resource group $RESOURCE_GROUP"
        echo_error "Please make sure the Key Vault was created successfully first."
        exit 1
    fi
    
    fix_keyvault_permissions
    verify_keyvault_access
}

# Run main function
main "$@"