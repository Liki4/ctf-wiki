# 反饋移位寄存器

一般的，一個 n 級反饋移位寄存器如下圖所示

![image-20180712201048987](./figure/n-fsr.png)

其中

- $a_0$，$a_1$，…，$a_{n-1}$，爲初態。
- F 爲反饋函數或者反饋邏輯。如果 F 爲線性函數，那麼我們稱其爲線性反饋移位寄存器（LFSR），否則我們稱其爲非線性反饋移位寄存器（NFSR）。
- $a_{i+n}=F(a_i,a_{i+1},...,a_{i+n-1})$ 。

一般來說，反饋移位寄存器都會定義在某個有限域上，從而避免數字太大和太小的問題。因此，我們可以將其視爲同一個空間中的變換，即

$(a_i,a_{i+1},...,a_{i+n-1}) \rightarrow (a_{i+1},...,a_{i+n-1},a_{i+n})$
.
對於一個序列來說，我們一般定義其生成函數爲其序列對應的冪級數的和。
