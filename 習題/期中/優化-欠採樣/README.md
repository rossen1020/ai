此程式碼是優化了參考的植物辨識程式碼, 雖然參考的資料有做過資料預處理,像是數據增強, 
但因為資料集是不平衡資料, 每一類植物擁有的數據都不一致,
這會引響我們在訓練模型時, 造成一些植物分類表現很好而一些不好, 為了解決這問題,我採用欠採樣去平衡資料
但因為此方法是透過將每類的數量去隨機選取至與最少數量的類相同,所以我透過採用合成樣本+特徵構建+PCA
這三種方法去進行數據擴大

(Ps. 因為檔案比較大, 我是在kaggle網站上進行訓練, 詳細內容我放在" 我自己的帳號 "<br/> 
* 請在[網址中](https://www.kaggle.com/code/mostytasen/undersampling)查看<br/>
建議看網站的會比較直觀,而且完整訓練過程都有在這裏面, 我還有做其他方法, 但因為效果差不多就沒放了,
<br/>
<br/>
以下是訓練過程的簡單解說:

---

原本全部資料集
![總類](https://github.com/rossen1020/ai/assets/99935090/d6e9a88b-0076-40a4-b01f-77493a2989c8)
先進行數據增強<br/>
![image](https://github.com/rossen1020/ai/assets/99935090/a9ab0afb-f375-4e74-85eb-959b5ffbf2b9)
<br/>透過欠採樣後, 將每項資料減至52項
![image](https://github.com/rossen1020/ai/assets/99935090/8060be63-7cf3-4acc-9c78-ffb5204cdae8)
<br/>再透過合成樣本+特徵構建+PCA將資料變多
![image](https://github.com/rossen1020/ai/assets/99935090/8961acd0-de30-4523-84b6-7d3fc24babdd)
<br/> 
### 運用Resnet50神經網路模型進行訓練, 得出結果
![image](https://github.com/rossen1020/ai/assets/99935090/ccf7f467-a160-41ab-8898-8098ab1bfcf2)
![image](https://github.com/rossen1020/ai/assets/99935090/41c3fa9a-9de7-4f27-b3c2-add359393c43)

