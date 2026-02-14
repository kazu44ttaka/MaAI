# 開発メモ

## Git リモート設定

本リポジトリは MaAI-Kyoto/MaAI のフォークです。リモートは以下のように設定しています。

| リモート名 | URL | 用途 |
|---|---|---|
| `origin` | https://github.com/kazu44ttaka/MaAI.git | 自分のフォーク（push先） |
| `upstream` | https://github.com/MaAI-Kyoto/MaAI | 本家リポジトリ（更新取得用） |

### よく使うコマンド

```bash
# 自分のフォークにプッシュ
git push origin main

# 本家の最新変更を取り込む
git fetch upstream
git merge upstream/main
```

## 別のマシンで同じ環境を構築する

```bash
# 1. 自分のフォークからクローン
git clone https://github.com/kazu44ttaka/MaAI.git
cd MaAI

# 2. 本家リポジトリを upstream として追加
git remote add upstream https://github.com/MaAI-Kyoto/MaAI

# 3. 確認
git remote -v
```
