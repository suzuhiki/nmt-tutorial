# nmt-tutrial
機械翻訳のチュートリアルの記録

コーパスなどの不用意な公開を防ぐため、作業リポジトリとは分けて公開用のリポジトリを作成しました。

パスなどが実行環境と異なる点をご理解ください。

# LSTMCellを用いたLSTM Seq2Seqの実装
英日対訳コーパス ASPEC(https://jipsti.jst.go.jp/aspec/) を用いて、英語から日本語への翻訳を実装しました。

実装にはtorch.LSTMCellを利用しています。
## ファイルの場所

| path | Description |
| --- | --- |
| `result/` | testデータを翻訳して出力した英語文 `lstm_s2s_{epoch数}_{epoch bleu(nltk)}`の書式|
| `script/python/` | 学習などに用いたスクリプトやクラスファイルが配置されている |
| `_train.ipynb` | モデルを学習するスクリプト |
| `_test.ipynb` | testデータを翻訳し、結果を書き出すスクリプト |

## 実験設定(チュートリアルで制限)
| パラメータ | 設定値 |
| --- | --- |
| 学習データ数 | 20,000文 |
| 学習データの最大文長 | 50 |
| バッチサイズ | 64 |
| エポック数 | 20と24についてmosesでBLEU値を計算 |
| Embeddingの次元数 | 256 |
| LSTMCellの隠れ層 | 256 |
| Optimizer | Adam |
| 学習率 | 0.001 |
| Dropout | 未実装 |

## 実験結果
### 学習中のValidationの値(nltkのsentence_bleuでSmoothingFunction.method1を追加して測定)
| 学習エポック数 | validation bleu値(平均値) | loss(総和) |
| --- | --- | --- |
| 1 | 0.017977688828171218 | 113910 |
| 4 | 0.022676879762375153 | 87659 |
| 8 | 0.025038486029513963 | 71316 |
| 12 | 0.025128757627956702 | 57692 |
| 16 | 0.02599041811858998 |46850 |
| 20 | 0.02585927372693223 | 38360 |
| 24 | 0.02667330542932727 | 31647 |

### testデータについてmosesでblue値を測定

| 学習エポック数 | test bleu値 | 
| --- | --- |
| 20 | 1.21, 21.8/3.4/0.4/0.1 | 
| 24 | 1.24, 22.5/3.4/0.4/0.1 | 
