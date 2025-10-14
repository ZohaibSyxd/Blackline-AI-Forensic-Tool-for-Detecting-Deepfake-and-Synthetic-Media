
# =========================
# 3) Fusion of TimeSformer + Xception, simple blend
# =========================
Write-Host "`n=== 3) Fusion of TimeSformer + Xception, simple blend ===`n" -ForegroundColor Cyan
& python backend/src/fuse_models.py `
  --xception    backend/data/derived/xception_scores_frames.jsonl `
  --timesformer backend/data/derived/timesformer_scores.jsonl `
  --manifest    backend/data/derived/manifest.csv `
  --agg-frame p95 --agg-clip median `
  --use-split --val-as-test `
  --out backend/data/derived/fusion_predictions.jsonl

# =========================
# 4) Fusion of TimeSformer + Xception, trained logistic regression
# =========================
Write-Host "`n=== 4) Fusion of TimeSformer + Xception, trained logistic regression ===`n" -ForegroundColor Cyan
& python backend/src/fuse_models_lr.py `
  --xception    backend/data/derived/xception_scores_frames.jsonl `
  --timesformer backend/data/derived/timesformer_scores.jsonl `
  --manifest    backend/data/derived/manifest.csv `
  --agg-frame p95 --agg-clip median `
  --use-split --val-as-test `
  --out backend/data/derived/fusion_predictions_lr.jsonl