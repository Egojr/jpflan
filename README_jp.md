# JPFlan

このリポジトリには、[jpflan-raw](https://huggingface.co/datasets/Ego/jpflan-raw)の生データを使用して、instruction tuning用データセットを作成するためのコードが含まれています。  
元の[Flan論文](https://arxiv.org/abs/2301.13688)では、1800以上の下流タスクデータセットが収集され、データセットで使用されている主要な言語が英語であり、データセットそれぞれに10の異なるテンプレートが用意されています。  
JPFlanは、Flanを模倣し日本語データセットであり、38種類の異なる高品質なオープンソースの日本語の下流タスクデータセットを収集し、データセットそれぞれに0-shot用の3つのテンプレート、few-shot用に1つのテンプレートを提供します。 

各データセットの情報、ライセンス、サイズ、内容は `dataset_info.csv` に記載されています。

### 使用方法
まず、リポジトリをコピーし、依存関係 (dependency) をインストールします：

```
git clone https://github.com/Egojr/jpflan.git
cd jpflan/
pip install -r requirements.txt
```

データセットを作成するためには、次のコマンドを実行します：

```
python generate_dataset.py
```

各データセットのサンプル数、few-shotのショット数、および最大長はカスタマイズ可能です。`templates.csv` の内容もファイル内にあるテンプレートのテキストを書き換えることで変更可能です。  
上記のコマンドを実行することで作成するデータセットは `./jpflan` フォルダに保存されます。  
このデータセットは、データセットのあるパスを指定して `datasets.load_from_disk` 関数を実行することで読み込むことができます。

### デフォルト設定のタスク分布
または、事前にデフォルト設定で作成されたデータセットを使用することもできます：[jpflan](https://huggingface.co/datasets/Ego/jpflan)。

デフォルト設定ではデータセットごとに0-shotとfew-shotを `min(10000, dataset_size)` でサンプリングします。つまり、データセットごとに最大20000サンプルまでを利用します。  
さらに、0-shotとfew-shot例に重複はありません。few-shotの場合は、本題がショットではないことも確認します。 

デフォルト設定で作成されたデータセットのタスク分布は次のとおりです:  
![Task Distribution](img/task_distribution_.png "Task Distribution")