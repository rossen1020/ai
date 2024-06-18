<P>此程式碼是優化了原本一開始的植物辨識程式碼, 因為我找的資料集是不平衡資料, 每一類植物擁有的數據都不一致,
這將會引響我們在訓練模型時, 造成一些植物分類表現很好一些不好, 而解決方法我採用欠採樣去平衡資料
但因為此方法是透過將每類的數量去隨機選取至與最少數量的類相同,所以我透過採用合成樣本+特徵構建+PCA
這三種方法去進行數據擴大</P>
<P> Ps. 因為檔案比較大, 我是在kaggle網站上進行訓練, 詳細內容請在(https://www.kaggle.com/code/mostytasen/undersampling)
查看, 完整訓練過程都有在這裏面, 我還有做其他方法, 但因為效果還好就沒放了
以下是訓練過程的簡單解說
</p> 

---

原本全部資料集
![總類](https://github.com/rossen1020/ai/assets/99935090/d6e9a88b-0076-40a4-b01f-77493a2989c8)
透過欠採樣後, 將每項資料減至52項
![image](https://github.com/rossen1020/ai/assets/99935090/8060be63-7cf3-4acc-9c78-ffb5204cdae8)
<br/>再透過合成樣本+特徵構建+PCA將資料變多
![image](https://github.com/rossen1020/ai/assets/99935090/8961acd0-de30-4523-84b6-7d3fc24babdd)
<br/>運用Resnet50神經網路模型進行訓練, 得出結果
![image](https://github.com/rossen1020/ai/assets/99935090/ccf7f467-a160-41ab-8898-8098ab1bfcf2)
![image](https://github.com/rossen1020/ai/assets/99935090/41c3fa9a-9de7-4f27-b3c2-add359393c43)

