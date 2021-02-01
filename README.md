## _Data Loading Pipline 최적화하여 GPU Util 99% 달성하기_

<br/>

## Author 👤 : **Yoonseo Kim**

[_Github_](https://github.com/myoons)  
[_Website_](https://ainote.tistory.com/)
<br/>
<br/>

## **설명**

_Cifar10 과 CelebA에 대해서 GPU Util 99%를 달성해보았습니다._  
_더욱 자세한 설명은 아래 링크에서 확인하실 수 있습니다._

_**[GPU Util 99% 달성하기](https://ainote.tistory.com/14)**_


<br/>
<br/>

## **사용한 방법들**

### **_1\. Multi Process Data Loading (Prefetch)_**  

<br/>

![1](https://user-images.githubusercontent.com/67945103/106479580-d516cc80-64ed-11eb-856c-330dfdef1b08.png)

<br/>

**_단일 프로세스이기에 생기는 1+2의 시간 지연을 없애자는 아이디어다. 즉, CPU 0이 한 Batch를 준비하여 GPU에 올려 학습하는 동안, 다른 CPU 1 (프로세스)가 다음 Batch를 준비하는 것이다._** 이렇게 되면 GPU는 시간의 지연 없이 바로 다음 Batch를 학습할 수 있기 때문에 지연이 일어나지 않는다. 이는 다음 Batch를 미리 가져오는 것이기 때문에 Prefetch라고 한다. 실제로는 다중의 프로세스들이 준비한 Batch를 공유하는 Queue에 넣고, 하나씩 빼서 GPU에 올리는 방법이다.  

<br/>


```python
from torch.utils.data.dataloader import DataLoader

train_loader = DataLoader(dataset=train_set,
                        num_workers=4,  # 사용할 Process의 수
                        batch_size=512,
                        persistent_workers=True)
```

<br/>

**_딥러닝 프레임워크 Pytorch를 사용하면 위와 같이 쉽게 구현할 수 있다. torch.utils.data.dataloader의 DataLoader 객체 중 num\_workers라는 인자가 있는데, 이를 1보다 크게 설정하면 Multi Process Data Loading이 구현된다. 참 쉽다._** 각자의 컴퓨터 상황에 맞춰 num\_workers의 수를 잘 조절하면 된다. 필자의 컴퓨터의 경우, 총 12개의 코어를 사용할 수 있어 최대 num\_workers를 12까지 늘릴 수 있다.

<br/>
<br/>

### **_2\. 크기가 작은 Datatype 사용하기_**

<br/>

![2](https://user-images.githubusercontent.com/67945103/106479583-d5af6300-64ed-11eb-8302-1b86dd90e020.png)

<br/>

Pytorch의 경우 모델의 Parameter은 Float32 Datatype을 가지고 있기에 Input 또한 Float32로 들어가야 한다. 하지만 Float32는 UINT8 (Unsigned Int 8로 0\~255까지의 숫자를 나타낼 수 있다)보다 4배나 그 크기가 크다. **_즉, 단적으로 숫자만 보자면 이는 데이터를 전송할 때 4배만큼의 시간이 더 걸린다는 뜻이다. 따라서 실제로 모델에 Input으로 넣기 전에는 크기가 작은 Datatype으로 가지고 있는 것이 전송 속도에 도움이 된다. 이미지의 경우 0\~255 사이의 UINT8로 가지고 있다가 모델에 넣어주기 직전에 Normalize 하여 Float32로 바꿔주는 것이 더 빠르다는 것이다._**

<br/>

위와 같은 방법들을 사용하면, 사용할 데이터가 모두 메모리에 올라가는 경우 GPU Util을 99% 찍을 수 있다. 필자는 Cifar10 데이터로 실험해 보았는데, 성공하였다. **_하지만 두 번째 경우, 즉 사용할 데이터가 모두 메모리에 올라가 지 않는 경우에는 위와 같은 방법들로만은 GPU Util을 99%을 달성할 수 없다._ _그 이유는 앞서 말했듯, 디스크에서 메모리로 데이터를 올리는 것이 너무 느리기 때문이다._**

<br/>

![3](https://user-images.githubusercontent.com/67945103/106479585-d647f980-64ed-11eb-9b93-e703fedd0c97.png)

<br/>

데이터가 모두 메모리에 올라가지 않을 때 관찰한 CPU 스탯이다. 위 사진의 빨간색과 아래 사진의 sy 값은 모두 CPU가 Kernel Mode로 돌아가는 비율을 의미한다. **_이 수치가 높은 이유는_** **_디스크에서 메모리로 데이터를 올리는 것이 너무 느리기 때문이다. 즉, 디스크에서 메모리로 데이터를 올리는 게 느려 계속 Request가 쌓이는 중이다._** 이러면 아직 디스크에서 메모리에 데이터가 올라오지 않아 Batch를 만들지 못하기에 우리의 중요한 목표 "GPU에서 한 Batch 학습이 끝나기 전에 다음 Batch 메모리에 준비하기"가 실패하게 된다.

<br/>

위와 같은 경우에는 다른 방법들을 추가로 사용하여 GPU Util을 올릴 수 있다. **_문제는 디스크에서 메모리로 데이터를 올리는 것이 너무 느리다는 것이었다. 그러면 디스크에서부터 데이터를 요청하는 횟수를 줄이면 된다. 그 방법은 데이터의 일부분을 메모리에 올려놓는 것이다. 이때 사용할 수 있는 것이 HDF5의 Chunk이다._**

<br/>
<br/>

### **_3\. Chunk Hit_**

<br/>

![4](https://user-images.githubusercontent.com/67945103/106479591-d7792680-64ed-11eb-937c-f55b958cf702.png)

<br/>

HDF5 (Hierarchical Data Format 5) 이란 HDF 그룹에 의해 관리되고 있는 대용량의 데이터를 저장하기 위한 파일 형식이다. 이름 그대로 계층적으로 구조화된 배열 데이터를 저장하기에 용이하다. Linux의 디렉터리 구조와 유사해 보인다.

<br/>

![5](https://user-images.githubusercontent.com/67945103/106479592-d811bd00-64ed-11eb-882b-7e64bdb4a182.png)

<br/>

Layout이란 다차원의 Dataset을 연속적인 File에 Mapping 시키는 방법을 말하는데, HDF5에는 Contiguous Layout과 Chunk Layout이 있다. 

<br/>

_1\. Contiguous Layout_

_\- Dataset을 일자로 편다._

_\- 배열의 메모리에 저장되는 방식과 유사하다._

_\- 한 개의 통으로 디스크에 저장된다._

<br/>

_2\. Chunk Layout_

_\- Dataset을 여러 개의 Chunk (블록)으로 나누어서 저장한다._

_\- 파일 안에 한 블록이 무작위로 저장된다._

_\- Chunk 별로 읽고 쓸 수 있다._

<br/>

```python
import h5py

celebA = h5py.File(DATA_DIR, 'w', rdcc_nslots=11213, rdcc_nbytes=1024**3,rdcc_w0=1)

celebA.create_dataset('images',
                    data=batch_images,
                    dtype=np.uint8,
                    chunks=(100, 3, 217, 178),  # 11 MB : Chunk Size
                    maxshape=(None, 3, 218, 178))
                                    
celebA.create_dataset('labels',
                    data=labels_h5[:size],
                    dtype=np.uint8,
                    chunks=(20000,))
```

<br/>

파이썬에서는_h5py_라이브러리로 HDF5 파일을 쉽게 다룰 수 있다. 더욱 자세한 내용은 아래 링크에서 확인할 수 있다.  

_[H5PY Document](https://docs.h5py.org/en/stable/high/file.html)_


<br/>

![6](https://user-images.githubusercontent.com/67945103/106479594-d811bd00-64ed-11eb-8106-2fdc81f0af90.png)

<br/>

Chunk를 사용하는 것이 GPU Util을 올리는데 중요한 이유는 위 사진이 그대로 설명해준다.

_1\. Chunk에 있는 데이터 하나를 참조하면, 해당 Chunk 전체가 메모리에 올라간다._

_2\. 이후 임의의 데이터를 참조했을 때, 해당 데이터가 메모리에 올라가 있는 Chunk에 있으면 메모리에서 바로 참조한다._

<br/>

따라서 필자가 _**Chunk Hit이라고**_ 한 이유는, 이 모양새가 Cache Hit과 유사하다고 생각했기 때문이다. **_Chunk가 메모리에 올라가 있는 것을 Chunk Cache라고 하는데,  Chunk Cache의 크기와 개수를 잘 조절하면 눈부신 Util 상승을 볼 수 있다._**

<br/>
<br/>

### _**4\. Batch Echoing**_

<br/>

![7](https://user-images.githubusercontent.com/67945103/106479596-d8aa5380-64ed-11eb-8405-20d708ef5596.png)

<br/>

**_Batch Echoing이란 GPU에 올라온 한 Batch를 여러 번 사용하는 것을 의미한다. "어 이렇게 되면 학습의 Randomness가 저해되는 것 아닌가요?" 정확하다. 그렇기에 본 방법은 명백한 득과 실이 존재한다. 득은 학습의 속도를 증가시킬 수 있다는 것과, 실은 Randomness가 감소한다는 것이다._**

<br/>

**_따라서 필자는 본 방법을 적용할 때 (그나마) Randomness를 유지하기 위하여 아래와 같은 트릭을 적용하였다._**

<br/>

_1\. 512개짜리 Batch를 GPU에 올림_

_2\. 512개짜리 Batch를 256개짜리 Batch 2개로 나눔 (이를 A, B라고 함)_

_3\. 모델이 A와 B 각각 학습_

_4\. 512개짜리 Batch의 순서를 섞음_

_5\. 섞인 512개짜리 Batch를 256개짜리 Batch 2개로 나눔 (이를 C. D라고 함)_

_6\. 모델이 C와 D를 각각 학습._

<br/>

위의 과정을 거쳐 한 Batch을 2번 사용하였다. 필자의 방법보다 훨씬 창의적이고 좋은 방법들이 많을 것이기에 각자의 상상력을 십분 발휘해보길 바란다. **_필자가 위와 같이 Batch Echoing을 적용한 이유는, 이렇게 되면 A, B, C, D 가 모두 다른 이미지들로 구성된 Batch가 되어 어느 정도 Randomness가 유지될 것이라고 생각했기 때문이다._**

<br/>

---

<br/>

## **배움**

사실 이번 포스팅에서 가져가야 할 것은 방법론적인 것들이 아니다. GPU Util이 떨어졌을 때 (최적화가 덜 되었을 때) 그 원인이 무엇인지 파악과 해결하는 능력과 방법이 중요하다. 그 프로세스는 아래와 같이 이루어진다.

<br/>

_1\. 하드웨어의 속도에 대한 감을 가지고, 프로세스가 느림을 인지_

_2\. 속도 저하의 원인을 파악_

이때는 여러가 도구를 사용할 수 있다. 예를 들어 CPU의 스탯을 확인할 수 있는 _top, htop, atop _등을 사용해 도움을 받을 수 있다. **_중요한 것은 무작정 원인을 파악하는 것이 아니라 가설을 세워야 한다는 것이다. 그 이유는 원인의 범위를 좁혀 효율적으로 파악할 수 있기 때문이다._**

_3\. 원인을 해결할 수 있는 방법 고안_

_4\. 실행_

<br/>

**_위와 같은 문제 해결 파이프라인의 이번 포스팅의 교훈이다. 다들 이것을 명심하고 즐거운 개발을 하길 바란다! _**요즘 회사에서 인턴을 하고 있는데, 이것이 너무 바빠 글을 쓰지 못하였다. 반성하며, 앞으로는 좀 더 자주 글을 올리고자 한다.

<br/>
<br/>

## **Reference**
[1] _**[Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)**_

[2] _**[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)**_

[3] _**[GPU Util 99% 달성하기](https://ainote.tistory.com/14)**_
