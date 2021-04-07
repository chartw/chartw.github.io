---
layout: post
title: "Lecture Summary 1"
subtitle: "ML Basics and Neural Networks"
categories: cs
tags: ml
comments: true  
# header-img: img/review/review-book-organize-thoughts-1.png
---
> `AI Application System` 강의 중 week 2~4 내용의 정리입니다.

# Machine Learning Basics

### ML Overview

Superviesd Learning(지도 학습) : 정답(label)을 제공한 뒤, 문제를 해결하라고 하는 것. 

Regression : data의 추정치를 근사하는 식을 만드는것이 목적

Classification : data를 구분하는 식을 만드는 것이 목적

Unsuperviesd Learning(비지도 학습) : label을 주지 않고, 컴퓨터에게 경향성을 찾아보라고 시키는 것.

Reinforcement Learing(강화 학습) : 환경에 대해 어떤 행동을 취하고, 이로부터 어떤 보상을 얻으면서 학습.

Machine Learning : 사람이 Feature extraction을 디자인

Deep Learning : 모든것을 자동으로 학습

Model Parameters : Learnable, 학습값

HyperParameters : User-defined, 유저 설정값

Data Split for ML : Data를 train / validation / test로 구분하여 사용 train : 교과서 / validation : 모의고사 / test : 수능

### Linear Regression

![/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled.png](/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled.png)

![/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%201.png](/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%201.png)

x : Inputs / y : Labels(Ground Truth) / θ : Parameters(weight) / bias : input데이터와 무관한 parameter

Loss function : 임의의 파라미터 θ에 의해 추정된 추정치 $\hat y$과 Ground Truth값 $y$의 차이를 측정하는 함수

Optimization : Loss function을 줄이는 방향으로 parameter에 feedback을 줌

### Nearest Neighbors

Intra-class variation : 클래스 내의 분포, 작을수록 밀집된 경우이므로 더 좋다.

Inter-class variation : 클래스 간의 거리, 클수록 클래스간에 구분이 명확하므로 더 좋다.

Nearest Neighbor : test 데이터를 가장 유사한(거리가 가까운)label의 class로 판단

Distance Metric : 거리를 계산하는 공식(L1, L2)

![/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%202.png](/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%202.png)

How fast training and prediction? : Train : O(1) but predict O(N) → BAD

k-Nearest Neighbor : 근접했던 k개 labels에 대해서 가장 많은 class로 추정

k값, Distance Metric과 같은것이 hyperparmeter이다.

Pixel distance는 학습에 유용하지 않기 때문에 nearest neighbor는 거의 사용되지 않는다.

### Dimensionality Reduction/Expansion

Curse of dimensionality : Dimension이 과하게 증가하면, 어느 순간 classifier 성능이 감소함(적당한 Dimension 선택 필요)

Dimension Reduction

PCA : 최적의 표현을 위해 최대 분산을 찾아 차원을 축소

LDA : class를 잘 구분하기 위해  차원을 축소

t-sne : 고차원 데이터를 저차원 데이터로 시각화 하는 기법

Dimension Expansion : Intra-class var은 크고, Inter-class var은 작아서 class 구분이 어려운경우, 차원을 확장시켜 분리함. (kernel trick)

### Linear Classifier

image x → f(x,W) → 3 Class Scores

W : Weights ex) ax+b=0(a,b : Weights). 학습에 따라 변화할수 있음. Parameter

Linear Classifier의 목표 : 

W를 잘 조정하여 정답 클래스의 score를 증가시키는 것(고양이 이미지일 경우, cat class의 score를 증가시키는 것)

= Loss Function을 줄이기 위해 W를 잘 조정하는것

Random Parameter Update → Feed-foward → Loss Function을 줄이기 위해 Optimization → Update...

![/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%203.png](/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%203.png)

32x32x3(3072)이미지 입력, 10개의 클래스 출력일 경우 : f(x,W) = Wx + b(bias)

f(x,W) : 10x1 / x : 3072x1 / W : 10x3072 / b : 10x1

Linear한 layer(W 행렬곱)만으로는 nonlinear한 경우 해결 불가. → layer 사이에 Nonlinearity(ex Sigmoid) activation function을 추가하여 해결 → Deep Neural Network의 기본 구조

### SVM Classifier vs Softmax Classifier

$L={1 \over N}\sum_iL_i(f(x_i,W),y_i)$

$L$ : dataset에 대한 전체 Loss / $L_i$ : Loss function

SVM Loss function(Hinge loss)

$L_i=\sum_{j\ne y_i} max(0,w_j^Tx_i-w_{yi}^Tx_i+\Delta)$

$\Delta$  : Margin(parameter)

$w_j^Tx_i-w_{yi}^Tx_i+\Delta<0$

즉 correct class score가 other class score보다 최소한 margin만큼 클때 이상적(Loss가 0).

Softmax Loss function(cross-entropy loss)

$P(Y=k|X=x_i)={e^sk\over \sum e^{s_j}}$

softmax function : score를 exp and normalize

$L_i=-logP(Y=y_i|X=x_i)$

cross-entropy loss : P() = correct class의 nomalize 값.  P() = 1.00 일 때만 Loss=0으로 이상적

SVM vs Softmax

SVM : margin 값만 넘기면 Loss = 0이므로 학습종료. 상대적으로 욕심이 적다.

Softmax : 정확히 correct class score가 1.00이 되어야 Loss = 0. 상대적으로 욕심이 많다.

# Neural Networks 1

### Traditional ML Approaches

기존에 linear classifier로 분리가 불가능한 dataset을  feature vector로 분리 가능하게 함

feature vector : 대상의 속성을 표현하는 방식 (Color Histogram, HoG)

전통적인 ML : 다양한 feature vector을 이어 붙여서 feature를 표현

### Introduction to Neural Networks

Tensor : 저차원부터 고차원까지 표현할수있는 일반적인 용어. 1d - Tensor, 6d - Tensor ...

activation function으로 ReLu를 사용하여, 2,3,n layer Neural Network를 구성 / ReLU : max(0,x)

ex) $f=W_2max(0,W_1x)$ : input - W1 - ReLU  - W2 - output (2-layer Neural Net or 1-hidden-layer Neural Net) hidden layer = max(0,W_1x)

### Gradient Descent

![/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%204.png](/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%204.png)

![/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%205.png](/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%205.png)

최선의 Weight W를 구하는 방법

 $w=w+\Delta w$ : w에 변화량을 update

$\Delta w_j=-\eta {\partial J\over \partial w_j}$ : gradient에 negative sign과 scaleing factor(learning rate)를 곱한값이 변화량

때문에 gradient가 0이 되는 방향(Loss fucntion이 0)으로 w가 변화(학습)함.

Numveric Gradient : 숫자를 대입해서 미분하는 것. 느리고, 변화량을 근사하였으므로 결국 근사값이다.

Analytic Gradient : 미분공식을 이용하여 한번에 계산(편미분 하는것). 빠르고, 정확하다.

### Computational Graph & Backpropagation

복잡한 Loss에 대해 gradient 계산 어려움 → Computational Graph + Backpropagation

![/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%206.png](/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%206.png)

![/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%207.png](/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%207.png)

Backpropagation : 끝에서부터 앞으로 Chain rule을 통해 모든 Gradient를 얻어낼 수 있음

Computational Graph : 연산을 graph화 하여 그리는것

Not unique! 묶어서 한번에 계산할수도 있음

### Regularization

$L(W)={1 \over N}\sum_{i=1}^NL_i(f(x_i,W),y_i)+\lambda R(W)$

Full loss = Data loss + Regularization loss

Data loss : Model prediction과 training data의 불일치를 측정(W, x에 의존적)

Regularization loss : training data에 overfitting을 예방 → simpler model(W에 의존적)

$\lambda$ = regularization strength(hyperparameter)

![/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%208.png](/assets/img/post_img/2021-04-07-Lecture Summary 1/Untitled%208.png)

# Neural Networks 2

### Data Preprocessing

weight optimize를 더 쉽게 하도록 전처리 하는 것.

zero-centered data : 데이터 평균 지점이 원점이 되도록. 자주 쓰임

normalized data : 각 축별로 scale이 맞춰지도록 함

decorrelated data : 각 축의 correlation을 없앰. othogonal하게

whitened data : 구의 분포(축에 대해 분산량이 동일한)

### Weight Initialization

Global Optimum(Minimum) : 원하는 최적 값 /  Local Minimum : Loss function이 non-convex할때 생기는 Global Optimum이 아닌 다른 지점들

가중치 초기화로 인한 문제점

작은 랜덤 값으로 가중치 초기화

W를 표준 정규분포에서 샘플링함. activation 함수(ex tanh)를 넣은 network에서는 작동하지 않음

W가 작은 값 → 입력인 X도 점점0으로 수렴 → gradient도 작아짐 → 업데이트가 일어나지 않게됨(Local Minimum)

좀더 큰 랜덤값으로 초기화

-1또는 1로 saturation → gradient 0 → Local Minimum

Xavier Initialization

표준 정규 분포를 입력 차원으로 수로 스케일링 해주는 것. $std = {1\over \sqrt{D_{in}}}$

ReLu 이용시, 음수 출력을 0 으로 만드므로, 분산이 절반이 되어 값이 너무 작아짐. 2를 추가로 곱하여 절반이 되지 않도록 함(Kaiming / MSRA Initialization)

### Stocahstic Gradient Descent(SGD)

많은 데이터를 샘플링 → 그것의 loss 구함  → Backprop, Update → 반복 → 전체의 평균을 추정하는 방법

Epoch : 전체의 데이터셋을 한바퀴 도는 것의 단위

(Mini)batch : 모든 데이터를 한번에 수행할수 없기 때문에, 수행할수 있는 범위 내에서 분할한 subset

Iteration : 1 epoch을 도는데 필요한 batch 수

### (Fancier) Optimizers

SGD의 문제점 : 결국 Local minima나 gradient가 0인 지점에 갇히게 되는 경우 발생 가능

SGD + Momentum : gradient와 velocity(속력)의 벡터 합으로 실제 변화량을 계산. gradient가 0이라도, velocity에 의해 계속 움직일수 있음

AdaGrad : gradient가 클경우는 완만하게, gradient가 작을경우는 급격하게 변화

Adam : Momentum과 AdaGrad의 hybrid 형태. 

### Regularization

Model Ensembles : overfitting 현상시 사용할수 있는 트릭. 유사한 여러 model을 학습 시킨후, 결과 확률의 평균중 최대값을 사용. 그러나 자원 사용이 많아 개인 레벨에서는 사용X

Regularization loss(Weight decay)을 이용하여 overfitting 해결 가능

일반적인 패턴 : training에서 randomness을 추가한뒤, testing에서 randomness를 평균화하여 추정

Dropout(feature level)

일정 확률로 노드를 꺼버림(weight 연결 끊어버림). 

feature간의 연결성을 낮춰, 분별력을 높임. 

확률에 의해 경우의 수가 늘어나므로, 변형된 모델을 사용하여 평균을 낸다는 점에서는 ensemble과 유사함.

test time : 모든 노드를 키고, score값에 노드의 확률 p를 곱하여 계산

Inverted dropout : test time에서 p를 곱하는것 대신, train time에서 p를 나눠줌

Data Augmentation

data를 실제 일어날법한 transformation을 적용하여 사용

Horizontal Flips : 좌우 대칭

Random crops and scales : random하게 patch를 잘라내서 training → (4 corners + center)*2(flip) = 10 patch로 testing

Color Jitter : fixel value(contrast, brightness)를 조절하여 training

Translation, Rotation, Scaling, Stretch, Shearing...

Automatic Data Augmentation

DropConnect : Dropout처럼 노드를 끄는게 아닌 특정 노드와 노드의 W 연결을 끊음

fixel level

Cutout : random한 영역을 0으로 바꿔 training (croping과 유사) 

Mixup : random으로 두 class를 blend한 이미지를 target score를 blend 비율로 label 정하여 training.

Cutmix : Cutout과 Mixup의 hybrid. Cutout한 영역에 다른 class를 mix. 이미지가 차지하는 영역에 비례하여 label 설정. 효과가 좋아 사용해볼만함

### Hyperparameter Tuning

Grid Search는 resolution이 커지면 최적 구간 놓칠수 있으므로, Random Search 주로 사용.

그러나 모든 구간을 Search하지 않고, lough하게 search한뒤, 결과가 좋은 일정 구간을 search하는 Coarse to Fine 기법 이용

Choosing Hyperparameters

1. Check initial loss : regulaization loss(weight decay)를 끄고, data loss만 체크
2. Overfit a small sample : 경향성을 보기 위해 small training data에 대해 100% accuracy를 만족하도록 overfitting(learning rate, weight initialization)
3. Find Learning Rate that makes loss go down : 전체 dataset에 대해 loss가 감소하는 learning rate 찾기. weight decay 킴. learning rate $1e^{-1}, 1e^{-2}, 1e^{-3}, 1e^{-4}$가 적정 값.
4. Coarse grid, train for ~1~5 epochs : 몇개의 learning rate와 weight decay값을 구한뒤, 1~5 epoch을 앞에서 구한 몇개의 모델에 training. weight decay  $1e^{-4}, 1e^{-5}, 0$가 적정 값.
5. Refine grid, train longer : 최선의 모델을 step4에서 선택 한뒤, fine하게 grid를 잡고, 10~20 epoch train.
6. Look at loss curves : Train / Val accuracy 커브가 overfitting인지, 정상인지 확인

    Train acc : batch 단위로 구해진 정확도

    Validation acc : 일정 구간마다 계산된 데이터 전체의 평균 정확도

    Overfitting : train은 증가하는데, val은 감소하는경우. regulaization을 증가시키거나, data 양 증가시켜 해결

    Underfitting : train과 val의 gap이 매우 작을 경우. model size를 키워야함.

7. GOTO setp 5 : curve를 보고, step5(4)로 돌아가서 재설정.