import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionFusion(nn.Module):
    """
    주의 메커니즘을 활용한 멀티모달 융합 모듈
    - 캔들스틱 이미지 특징
    - 뉴스 감성 특징
    - 가격 데이터 특징 
    을 융합하여 최종 거래 신호를 생성
    """
    def __init__(self, feature_dim=64):
        super(AttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        
        # 각 모달리티별 특징 추출기
        self.candlestick_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.price_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 주의 가중치 생성기
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 3, 3),
            nn.Softmax(dim=-1)
        )
        
        # 최종 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Tanh()  # -1 ~ 1 사이의 거래 신호 생성
        )
    
    def forward(self, candlestick_signal, sentiment_signal, price_signal):
        # 각 신호를 텐서로 변환
        if not isinstance(candlestick_signal, torch.Tensor):
            candlestick_signal = torch.tensor([candlestick_signal], dtype=torch.float32).unsqueeze(1)
        if not isinstance(sentiment_signal, torch.Tensor):
            sentiment_signal = torch.tensor([sentiment_signal], dtype=torch.float32).unsqueeze(1)
        if not isinstance(price_signal, torch.Tensor):
            price_signal = torch.tensor([price_signal], dtype=torch.float32).unsqueeze(1)
        
        # 각 모달리티별 특징 추출
        candlestick_feature = self.candlestick_encoder(candlestick_signal)
        sentiment_feature = self.sentiment_encoder(sentiment_signal)
        price_feature = self.price_encoder(price_signal)
        
        # 모든 특징 결합
        combined_features = torch.cat([candlestick_feature, sentiment_feature, price_feature], dim=1)
        
        # 주의 가중치 생성
        attention_weights = self.attention(combined_features.view(-1, self.feature_dim * 3))
        
        # 주의 가중치 적용
        weighted_candlestick = candlestick_feature * attention_weights[:, 0].unsqueeze(1)
        weighted_sentiment = sentiment_feature * attention_weights[:, 1].unsqueeze(1)
        weighted_price = price_feature * attention_weights[:, 2].unsqueeze(1)
        
        # 가중 특징 결합
        weighted_features = torch.cat([weighted_candlestick, weighted_sentiment, weighted_price], dim=1)
        
        # 최종 융합 신호 생성
        fusion_signal = self.fusion_layer(weighted_features)
        
        return fusion_signal.item(), attention_weights.detach().cpu().numpy()[0]

class TransformerFusion(nn.Module):
    """
    트랜스포머 기반 멀티모달 융합 모듈
    - 자기주의(self-attention) 메커니즘으로 모달리티 간 관계 학습
    """
    def __init__(self, feature_dim=64, num_heads=4):
        super(TransformerFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # 각 모달리티별 특징 추출기
        self.candlestick_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        self.price_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        # 멀티헤드 어텐션
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 피드포워드 네트워크
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 최종 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Tanh()  # -1 ~ 1 사이의 거래 신호 생성
        )
    
    def forward(self, candlestick_signal, sentiment_signal, price_signal):
        # 각 신호를 텐서로 변환
        if not isinstance(candlestick_signal, torch.Tensor):
            candlestick_signal = torch.tensor([candlestick_signal], dtype=torch.float32).unsqueeze(1)
        if not isinstance(sentiment_signal, torch.Tensor):
            sentiment_signal = torch.tensor([sentiment_signal], dtype=torch.float32).unsqueeze(1)
        if not isinstance(price_signal, torch.Tensor):
            price_signal = torch.tensor([price_signal], dtype=torch.float32).unsqueeze(1)
        
        # 각 모달리티별 특징 추출
        candlestick_feature = self.candlestick_encoder(candlestick_signal)
        sentiment_feature = self.sentiment_encoder(sentiment_signal)
        price_feature = self.price_encoder(price_signal)
        
        # 시퀀스 생성 (각 모달리티를 시퀀스의 토큰으로 간주)
        # [배치, 시퀀스 길이, 특징 차원]
        sequence = torch.cat([
            candlestick_feature.unsqueeze(1),
            sentiment_feature.unsqueeze(1),
            price_feature.unsqueeze(1)
        ], dim=1)  # [batch, 3, feature_dim]
        
        # 자기주의 메커니즘 적용
        attn_output, _ = self.multihead_attention(sequence, sequence, sequence)
        
        # 잔차 연결
        attn_output = attn_output + sequence
        
        # 피드포워드 네트워크
        ff_output = self.feed_forward(attn_output)
        
        # 잔차 연결
        output = ff_output + attn_output
        
        # 평균 풀링으로 시퀀스 차원 제거
        pooled_output = output.mean(dim=1)  # [batch, feature_dim]
        
        # 최종 융합 신호 생성
        fusion_signal = self.fusion_layer(pooled_output)
        
        return fusion_signal.item()

# 모델 사용 예시
def test_fusion_models():
    # 예시 신호
    candlestick_signal = 0.7  # 캔들스틱 패턴 신호 (예: ResNet50 출력)
    sentiment_signal = 0.3    # 뉴스 감성 신호
    price_signal = 0.1        # 가격 변화율
    
    # 주의 메커니즘 모델
    attention_model = AttentionFusion()
    attention_signal, attention_weights = attention_model(
        candlestick_signal, sentiment_signal, price_signal
    )
    
    print(f"주의 메커니즘 융합 신호: {attention_signal:.4f}")
    print(f"주의 가중치: 캔들스틱={attention_weights[0]:.2f}, 감성={attention_weights[1]:.2f}, 가격={attention_weights[2]:.2f}")
    
    # 트랜스포머 모델
    transformer_model = TransformerFusion()
    transformer_signal = transformer_model(
        candlestick_signal, sentiment_signal, price_signal
    )
    
    print(f"트랜스포머 융합 신호: {transformer_signal:.4f}")

if __name__ == "__main__":
    test_fusion_models() 