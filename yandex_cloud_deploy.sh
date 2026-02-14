#!/bin/bash

#############################################################################
# EchoFlow 2.0 - Yandex Cloud Automated Deployment Script
#############################################################################
# 
# This script automates the entire deployment process on Yandex Cloud:
# - Creates GPU VM instance
# - Installs all dependencies
# - Clones repository
# - Starts training
#
# Usage:
#   ./yandex_cloud_deploy.sh
#
#############################################################################

set -e

echo "======================================================================"
echo "  EchoFlow 2.0 - Yandex Cloud Deployment"
echo "======================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VM_NAME="echoflow-training"
ZONE="ru-central1-a"
PLATFORM="gpu-standard-v1"
CORES=8
MEMORY=96
GPUS=1
DISK_SIZE=100
IMAGE_FAMILY="ubuntu-2204-lts"

#############################################################################
# Check Prerequisites
#############################################################################

echo "Checking prerequisites..."

# Check if yc CLI is installed
if ! command -v yc &> /dev/null; then
    echo -e "${RED}✗ Yandex CLI (yc) not found${NC}"
    echo ""
    echo "Please install Yandex CLI:"
    echo "  curl https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Yandex CLI found${NC}"

# Check if authenticated
if ! yc config list &> /dev/null; then
    echo -e "${RED}✗ Not authenticated${NC}"
    echo ""
    echo "Please authenticate:"
    echo "  yc init"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Authenticated${NC}"

# Get SSH key
if [ ! -f ~/.ssh/id_rsa.pub ]; then
    echo -e "${YELLOW}⚠ No SSH key found, generating...${NC}"
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
    echo -e "${GREEN}✓ SSH key generated${NC}"
else
    echo -e "${GREEN}✓ SSH key found${NC}"
fi

echo ""

#############################################################################
# Create VM
#############################################################################

echo "======================================================================"
echo "  Creating GPU VM Instance"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  Name:     $VM_NAME"
echo "  Zone:     $ZONE"
echo "  Platform: $PLATFORM"
echo "  GPU:      $GPUS x Tesla V100"
echo "  vCPU:     $CORES"
echo "  RAM:      ${MEMORY} GB"
echo "  Disk:     ${DISK_SIZE} GB SSD"
echo ""
echo "Estimated cost: ~₽50/hour (~₽900 for 18 hours)"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Creating VM..."

# Check if VM already exists
if yc compute instance get $VM_NAME &> /dev/null; then
    echo -e "${YELLOW}⚠ VM '$VM_NAME' already exists${NC}"
    read -p "Delete and recreate? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing VM..."
        yc compute instance delete $VM_NAME --async
        sleep 10
    else
        echo "Using existing VM..."
        VM_IP=$(yc compute instance get $VM_NAME --format json | jq -r '.network_interfaces[0].primary_v4_address.one_to_one_nat.address')
        echo -e "${GREEN}✓ VM IP: $VM_IP${NC}"
        echo ""
        echo "======================================================================"
        echo "  Connecting to VM"
        echo "======================================================================"
        echo ""
        ssh -o StrictHostKeyChecking=no ubuntu@$VM_IP
        exit 0
    fi
fi

# Create VM
yc compute instance create \
  --name $VM_NAME \
  --zone $ZONE \
  --platform $PLATFORM \
  --cores $CORES \
  --memory $MEMORY \
  --gpus $GPUS \
  --network-interface subnet-name=default-$ZONE,nat-ip-version=ipv4 \
  --create-boot-disk image-folder-id=standard-images,image-family=$IMAGE_FAMILY,size=$DISK_SIZE,type=network-ssd \
  --ssh-key ~/.ssh/id_rsa.pub \
  --async

echo ""
echo "Waiting for VM to start..."
sleep 30

# Get VM IP
VM_IP=$(yc compute instance get $VM_NAME --format json | jq -r '.network_interfaces[0].primary_v4_address.one_to_one_nat.address')

if [ -z "$VM_IP" ] || [ "$VM_IP" == "null" ]; then
    echo -e "${RED}✗ Failed to get VM IP${NC}"
    exit 1
fi

echo -e "${GREEN}✓ VM created successfully${NC}"
echo -e "${GREEN}✓ VM IP: $VM_IP${NC}"
echo ""

# Wait for SSH
echo "Waiting for SSH to be ready..."
for i in {1..30}; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$VM_IP "echo 'SSH ready'" &> /dev/null; then
        echo -e "${GREEN}✓ SSH ready${NC}"
        break
    fi
    echo "  Attempt $i/30..."
    sleep 10
done

echo ""

#############################################################################
# Deploy and Start Training
#############################################################################

echo "======================================================================"
echo "  Deploying EchoFlow 2.0"
echo "======================================================================"
echo ""

# Create deployment script
cat > /tmp/deploy_echoflow.sh << 'EOF'
#!/bin/bash
set -e

echo "======================================================================"
echo "  Installing Dependencies"
echo "======================================================================"
echo ""

# Update system
sudo apt-get update -qq

# Install NVIDIA drivers and CUDA
echo "Installing NVIDIA drivers..."
sudo apt-get install -y -qq nvidia-driver-535 nvidia-cuda-toolkit

# Install Python dependencies
echo "Installing Python packages..."
sudo apt-get install -y -qq python3-pip git

# Clone repository
echo ""
echo "======================================================================"
echo "  Cloning Repository"
echo "======================================================================"
echo ""

cd /home/ubuntu
if [ -d "Voice" ]; then
    rm -rf Voice
fi

git clone https://github.com/izoomlentoboy-creator/Voice.git
cd Voice

echo ""
echo "======================================================================"
echo "  Starting Training"
echo "======================================================================"
echo ""
echo "Training will take approximately 18-20 hours."
echo "You can disconnect and reconnect later."
echo ""
echo "To monitor progress:"
echo "  ssh ubuntu@$(curl -s ifconfig.me)"
echo "  tail -f /tmp/training.log"
echo ""

# Make script executable
chmod +x train_ultimate.sh

# Start training in screen
sudo apt-get install -y -qq screen
screen -dmS training bash -c './train_ultimate.sh 2>&1 | tee /tmp/training.log'

echo ""
echo "✅ Training started in background!"
echo ""
echo "To attach to training session:"
echo "  screen -r training"
echo ""
echo "To detach: Press Ctrl+A, then D"
echo ""
EOF

# Copy and execute deployment script
scp -o StrictHostKeyChecking=no /tmp/deploy_echoflow.sh ubuntu@$VM_IP:/tmp/
ssh -o StrictHostKeyChecking=no ubuntu@$VM_IP "bash /tmp/deploy_echoflow.sh"

echo ""
echo "======================================================================"
echo "  Deployment Complete!"
echo "======================================================================"
echo ""
echo "VM Information:"
echo "  Name: $VM_NAME"
echo "  IP:   $VM_IP"
echo ""
echo "To connect:"
echo "  ssh ubuntu@$VM_IP"
echo ""
echo "To monitor training:"
echo "  ssh ubuntu@$VM_IP 'tail -f /tmp/training.log'"
echo ""
echo "To attach to training session:"
echo "  ssh ubuntu@$VM_IP"
echo "  screen -r training"
echo ""
echo "⚠️  IMPORTANT: Stop VM after training to avoid charges!"
echo "  yc compute instance stop $VM_NAME"
echo "  yc compute instance delete $VM_NAME"
echo ""
echo "======================================================================"
