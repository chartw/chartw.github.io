---
layout: post
title: "VScode 디버깅 배열 조사식"
subtitle: "VScode 조사식"
categories: devenv
tags: 
comments: true  
# header-img: img/review/review-book-organize-thoughts-1.png

---


> VScode에서 mingw로 디버깅시, 배열에 대한 조사식이 제대로 적용되지 않았다.

Visual Studio 에서 디버깅시 배열의 조사식으로 `arr, 10` 을 입력하여 arr[10]의 원소를 확인할 수 있다.

하지만 VScode의 경우, 위와 같이 조사식을 설정해도 확인할 수 없다. 이를 해결하기 위해 조사식을 다음과 같이 입력한다.

```cpp
*(int(*)[10])arr
```

![/assets/img/post_img/2021-04-27-devenv--VScodewatcharr/Untitled.png](/assets/img/post_img/2021-04-27-devenv--VScodewatcharr/Untitled.png)

위와 같이 배열의 원소가 출력되는 것을 확인하였다.