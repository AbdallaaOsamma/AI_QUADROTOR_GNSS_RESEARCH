import torch
import torch.nn as nn

class NavigationCNN(nn.Module):
    """
    CNN for vision-based navigation.
    Input: grayscale image (1, H, W)
    Output: [vx, vy, vz, rz] commands
    """
    def __init__(self, img_height=84, img_width=84, num_outputs=4):
        super(NavigationCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv1: 1x84x84 -> 32x42x42
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv2: 32x42x42 -> 64x21x21
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv3: 64x21x21 -> 128x10x10
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv4: 128x10x10 -> 256x5x5
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate flattened size: 256 * 5 * 5 = 6400
        conv_output_size = 256 * 5 * 5
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
    
    def forward(self, x):
        # x shape: (batch, 1, H, W)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x