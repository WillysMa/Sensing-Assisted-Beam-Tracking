#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact:mamengyuan410@gmail.com
@file: MyFuncs.py
@time: 2025/3/10 13:42
"""
from pytorch_model_summary import summary
import torch.nn.functional as F
import torch.nn as nn
import torch


class ImageFeatureExtractor(nn.Module):
    def __init__(self, n_feature, in_channel=1):
        super(ImageFeatureExtractor, self).__init__()


        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=4, kernel_size=(3, 3), stride=1,padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=0),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1) ,padding=0),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=0),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=0),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=0),
            nn.MaxPool2d(kernel_size=(2, 2))
            # nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(2),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.channel_attention = False
        self.spatial_attention = False

        if self.channel_attention:
            # Channel Attention Module
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(64, 64 // 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64 // 2, 64, kernel_size=1),
                nn.Sigmoid()
            )
        
        if self.spatial_attention:
            # Spatial Attention Module
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            )

        # 全局平均池化层
        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # 全连接层用于减少特征维度
        self.fc_layer = nn.Sequential(
             nn.Linear(64 * 7 * 7, 512),
             nn.ReLU(),
             nn.Dropout(0.5),
             nn.Linear(512, 128),
             nn.ReLU(),
             nn.Dropout(0.3),
             nn.Linear(128, 64),
             nn.ReLU(),
             nn.Dropout(0.2),
             nn.Linear(64, n_feature)
        )

        # self.fc2 = nn.Linear(256, 64)
        # self.fc_layer = nn.Linear(32, n_feature)


    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        spatial_features = []

        # 对每个时间步分别处理
        for t in range(seq_length):
            frame = x[:, t, :, :,:]  # 获取第t个时间步的帧
            
            # Apply CNN layers
            frame_features = self.cnn_layers(frame)  # 应用2D CNN
            
            # Apply channel attention
            if self.channel_attention:
                channel_att = self.channel_attention(frame_features)
                frame_features = frame_features * channel_att
            
            # Apply spatial attention
            if self.spatial_attention:
                spatial_att = self.spatial_attention(frame_features)  # Generate attention map
                frame_features = frame_features * spatial_att  # Apply attention
            
            # Flatten and process through FC layers
            frame_features = self.flatten(frame_features)
            frame_features = self.fc_layer(frame_features)
            
            spatial_features.append(frame_features)

        # 将所有时间步的特征拼接在一起
        spatial_features = torch.stack(spatial_features, dim=1)  # 形状: (batch_size, seq_length, n_feature)
        return spatial_features



class ImageModalityNet(nn.Module):
    def __init__(self, feature_size, num_classes, gru_params):
        super(ImageModalityNet, self).__init__()
        '''
        This model uses only image or radar as input for learning.
        image=True indicates the use of image data; image=False implies using radar data.
        '''
        self.name = 'ImageModalityNet'
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        assert gru_input_size == feature_size, f"Error: gru_input_size ({gru_input_size}) must be equal to feature_size ({feature_size})"

        self.feature_extraction = ImageFeatureExtractor(feature_size) # image input only

  
        self.GRU = nn.GRU(input_size=gru_input_size, hidden_size=gru_hidden_size, num_layers=gru_num_layers,
                          dropout=0.8, batch_first=True)

        self.Temporal_attention = False

        if self.Temporal_attention:
            # Temporal attention module
            self.temporal_attention = nn.Sequential(
                nn.Linear(gru_hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        # Add LayerNorm before GRU input
        self.layer_norm = nn.LayerNorm(gru_input_size)
        # Classifier
        # self.classifier = nn.Linear(gru_hidden_size, num_classes)
        self.classifier = nn.Sequential(
             nn.Linear(gru_hidden_size, 64),
             nn.ReLU(),
             nn.Dropout(0.5),
             nn.Linear(64, 64),
             nn.ReLU(),
             nn.Dropout(0.3),
             nn.Linear(64, num_classes)
        )

        # self.embed = nn.Embedding(num_classes, num_classes)


    def forward(self, image_batch, beam):
        batch_size, seq_len, _, _, _ = image_batch.size()
        # Extract features using the feature extraction network
        # beam = self.embed(beam).unsqueeze(1)


        features = self.feature_extraction(image_batch)

        # Apply LayerNorm to the features
        features = self.layer_norm(features)
        Seq_out, _ = self.GRU(features)

        # Apply temporal attention
        if self.Temporal_attention:
            attn_weights = self.temporal_attention(Seq_out)
            attn_weights = F.softmax(attn_weights, dim=1)
        
            # Apply attention weights to sequence output
            # This creates a weighted sum across the time dimension
            context_vector = torch.sum(Seq_out * attn_weights, dim=1)
            
            # Expand context vector to match sequence length for residual connection
            context_vector_expanded = context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        
            # Combine with original sequence using residual connection
            enhanced_seq_out = Seq_out + context_vector_expanded
        else:
            enhanced_seq_out = Seq_out

        # beam_tmp = beam.repeat(1,seq_len,1)
        # CatFeature = torch.cat((Seq_out, beam_tmp), dim=2)

        Pred = self.classifier(enhanced_seq_out) # Final classification layer

        return Pred, features, _


class StudentModalityNet(nn.Module):
    def __init__(self, feature_size, num_classes, gru_params):
        super(StudentModalityNet, self).__init__()
        '''
        This model uses only image or radar as input for learning.
        Minimal CNN architecture with GlobalAveragePool and GlobalMaxPool.
        '''
        self.name = 'StudentModalityNet'
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        assert gru_input_size == feature_size, f"Error: gru_input_size ({gru_input_size}) must be equal to feature_size ({feature_size})"

        # Minimal CNN layers
        self.cnn_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Third conv block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(128), #  (128, 7, 7)
            nn.ReLU(inplace=True),
        )
        
        # Global pooling layers
        # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Feature fusion and projection
        self.fc_layer = nn.Sequential(
            nn.Linear(128, 64),  
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, feature_size)
        )

        self.GRU = nn.GRU(input_size=gru_input_size, hidden_size=gru_hidden_size, num_layers=gru_num_layers,
                          dropout=0.8, batch_first=True)

        # Add LayerNorm before GRU input
        self.layer_norm = nn.LayerNorm(gru_input_size)
        
        # Classifier
        self.classifier = nn.Sequential(
             nn.Linear(gru_hidden_size, 64),
             nn.ReLU(),
             nn.Dropout(0.5),
             nn.Linear(64, num_classes)
        )

    def forward(self, image_batch, beam):
        batch_size, seq_len, channels, height, width = image_batch.size()
        
        # Process each frame in the sequence
        features_list = []
        for t in range(seq_len):
            frame = image_batch[:, t, :, :, :]  # Shape: (batch_size, 1, 224, 224)
            

            # Apply CNN layers
            cnn_features = self.cnn_layers(frame)  # Shape: (batch_size, 128, H', W')
            
            # Apply global pooling
            # avg_pooled = self.global_avg_pool(cnn_features)  # Shape: (batch_size, 128, 1, 1)
            max_pooled = self.global_max_pool(cnn_features)  # Shape: (batch_size, 128, 1, 1)
            
            # Flatten and concatenate
            # avg_pooled = avg_pooled.view(batch_size, -1)  # Shape: (batch_size, 128)
            pooled_features = max_pooled.view(batch_size, -1)  # Shape: (batch_size, 128)
            # pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)  # Shape: (batch_size, 256)
            
            # Project to desired feature size
            frame_features = self.fc_layer(pooled_features)  # Shape: (batch_size, feature_size)
            features_list.append(frame_features)
        
        # Stack features for all time steps
        features = torch.stack(features_list, dim=1)  # Shape: (batch_size, seq_len, feature_size)
        
        # Apply LayerNorm to the features
        features_norm = self.layer_norm(features)
        
        # GRU processing
        Seq_out, _ = self.GRU(features_norm)

        # Final classification
        Pred = self.classifier(Seq_out)

        return Pred, features, Seq_out
    

class LinearMapping(nn.Module):
    def __init__(self, teacher_seq_length, std_seq_length):
        super(LinearMapping, self).__init__()
        self.teacher_seq_length = teacher_seq_length  # N (number of teacher features)
        self.std_seq_length = std_seq_length  # M (number of student features)

        
        # Linear mapping weights: M x N learnable parameters for each feature dimension
        # Shape: (output_seq_length, input_seq_length)
        # This allows each output feature to be a weighted combination of all input features
        self.weight = nn.Parameter(torch.randn(teacher_seq_length, std_seq_length))
        
        # Initialize weights with Xavier uniform initialization
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        
        x_transposed = x.transpose(1, 2)
        
        output = torch.matmul(x_transposed, self.weight)
        output_rearranged = output.transpose(1, 2)
        
        return output_rearranged


if __name__ == "__main__":

    # 使用例子
    # n_feature = 64
    # in_channel = 16 # 假设你想要提取128个特征
    # model = RadarFeatureExtractor(n_feature=n_feature, in_channel=in_channel)
    #
    # # 输入数据形状: (batch_size, sequence_length, channels, height, width)
    # input_data = torch.randn(8, 5, 16, 64, 64)  # 示例数据
    # output_features = model(input_data)
    # print(output_features.shape)  # 应该输出torch.Size([10, 5, 128])

    # Example usage:
    FEATURE_SIZE = 64  # 假设每种模态的特征大小为128
    GRU_PARAMS =  (FEATURE_SIZE*2, 64, 1) # GRU_INPUT_SIZE, GRU_HIDDEN_DIM, GRU_NUM_LAYERs
    NUM_CLASS = 64  # 分类数
    BATCH_SIZE = 5
    SEQ_LENGTH = 7



    # 输入数据形状: (batch_size, seq_length, feature_size)
    image_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, 1, 224,224)  # 图像特征
    radar_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, 16,64,64)  # 雷达特征
    beam = torch.randint(low=0, high=NUM_CLASS, size=(BATCH_SIZE, SEQ_LENGTH), dtype=torch.long)

    model = ImageModalityNet(FEATURE_SIZE, NUM_CLASS, GRU_PARAMS)
    print(summary(model, image_input, radar_input, beam[:,0], show_input=True, show_hierarchical=True))

    # Summarize and print all trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal number of trainable parameters: {trainable_params}")

    # resnet18 = resnet18_mod(pretrained=True, progress=True, num_classes=FEATURE_SIZE)
    # resnet18_state_dict = resnet18.state_dict()
    # # print(resnet18_state_dict)
    # # Create ResNet6
    #
    #
    # # print(model)
    # # Transfer weights
    # model_state_dict = model.state_dict()
    # # print(resnet6_state_dict)
    # # Step 3: Extract and map pretrained weights for layer1 and layer2
    # for name, param in resnet18_state_dict.items():
    #     # Only copy weights for layer1 and layer2
    #     if name.startswith("layer1.0") or name.startswith("layer2.0") or name.startswith('layer3.0') or name.startswith(
    #             'layer4.0'):
    #         if name in model_state_dict:  # Ensure the key exists in ResNet6
    #             model_state_dict[name] = param
    #
    # # Step 4: Load the filtered state_dict into ResNet6
    # model.load_state_dict(model_state_dict)

    # Load the pre-trained weights for the entire AutoEncoder
    # autoencoder_path = './encoder.pth'  # Replace with the actual path to your AutoEncoder model file
    # autoencoder_state_dict = torch.load(autoencoder_path,map_location='cpu')
    # model.feature_extraction.image_encoder.load_state_dict(autoencoder_state_dict)
    # Print parameter names and their shapes
    # for param_name, param_tensor in autoencoder_state_dict.items():
    #     print(f"Layer: {param_name} | Shape: {param_tensor.shape}")
    # Load the extracted encoder weights into the model's AutoEncoder


    output = model(image_input, radar_input, beam[:,0])
    print(output.shape)  # 应该输出torch.Size([8, 10]) 如果是10分类问题的话