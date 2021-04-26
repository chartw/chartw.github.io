---
layout: post
title: "VScode 출력 한글 깨짐"
subtitle: "VScode 환경 설정"
categories: devenv
tags: 
comments: true  
# header-img: img/review/review-book-organize-thoughts-1.png

---


> vscode로 한글 출력시 깨지는 현상이 발생하였다.

이를 해결하기 위해서는 다음과 같은 방법이 있다.

### 인코딩 변경

대부분은 utf-8 인코더를 사용한다. 하지만 vscode에서는 utf-8로 인코딩된 한글을 출력할 수 없다. 때문에 가장 간편한 방법은 인코더를 euc-kr로 변경하는 것이다.

![/assets/img/post_img/2021-04-27-devenv--VScodeopkor/Untitled.png](/assets/img/post_img/2021-04-27-devenv--VScodeopkor/Untitled.png)

우측 하단의 utf-8을 눌러

![/assets/img/post_img/2021-04-27-devenv--VScodeopkor/Untitled%201.png](/assets/img/post_img/2021-04-27-devenv--VScodeopkor/Untitled%201.png)

위의 작업 선택 창에서 `인코딩하여 다시 열기`를 누른 후, euc-kr을 선택하여 변경한다.

이후 다시 컴파일 하여 실행하면, 정상적으로 실행되는 것을 볼수 있다.

하지만 이것은 euc-kr 인코더의 사용을 강제하므로, utf-8 인코더를 사용하면서 한글을 출력할 수 있는 다른 방법이 존재한다.

### 레지스트리

vscode에서 프로그램 컴파일 - 실행 시(ex c_mingw)기본적으로 cmd를 이용하여 출력된다.

cmd에서 chcp 명령어를 이용해 현재 활성화된 code page를 확인할 수 있다.

따로 설정을 건드리지 않았다면, 949로 설정되어 있을 것이다.

```powershell
chcp
Active code page: 949
```

이것은 인코더를 949 사용한다는 의미이다. 때문에 이것을 utf-8인 65001로 변경하여야 한다. 변경을 위한 명령어는 다음과 같다.

```powershell
chcp 65001
```

하지만 이 명령어는 일시적이다. cmd를 재부팅 하면 설정값이 초기화 된다. 

때문에 기본값을 65001로 설정해야만 한다.

레지스트리를 수정하여 기본값을 변경할 수 있다.

`win+r → regedit` 으로 레지스트리 편집기에 들어간다.

수정할 레지스트리의 경로는 다음과 같다.

`컴퓨터\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Nls\CodePage`

![/assets/img/post_img/2021-04-27-devenv--VScodeopkor/Untitled%202.png](/assets/img/post_img/2021-04-27-devenv--VScodeopkor/Untitled%202.png)

CodePage에서 OEMCP를 선택한 뒤, 값을 65001로 변경한다.

![/assets/img/post_img/2021-04-27-devenv--VScodeopkor/Untitled%203.png](/assets/img/post_img/2021-04-27-devenv--VScodeopkor/Untitled%203.png)

재부팅 한 뒤, chcp로 확인해보면 65001로 변경되어있다. 이 상태에서 vscode로 한글을 출력하면 정상적으로 출력된다.