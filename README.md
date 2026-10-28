# Music Genre Classification and Taste-Based Recommender

Classify songs into genres from audio features, build a listening profile from your own playlist, and get song suggestions from the catalog and from live 30-second previews online.

Everything lives in one script, `music_pipeline.py`. It started as a course project on genre classification (feature table, EDA, model comparison, confusion matrix, a cosine-similarity recommender) and grew from there.

---

## What it does

1. **Loads the GTZAN feature CSV** (1,000 songs, 10 genres) and adds 38 engineered features on top of the 57 that ship with the dataset.
2. **Compares 8 classifiers** with cross-validation that never puts the same song in both train and test, then combines the best four into a soft-voting ensemble.
3. **Builds a taste embedding**: a supervised timbre space plus a valence/arousal estimate. Similar songs are neighbours in this space.
4. **Profiles a playlist**: genre mix, diversity, tempo, mood clusters.
5. **Recommends** in three ways:
   - nearest songs in the GTZAN catalog,
   - a gradual mood path through the catalog (iso-principle),
   - live suggestions: it searches the iTunes India store, downloads 30-second previews, analyses the real audio, and ranks them against your playlist.

---

## Quick start

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn librosa joblib imageio-ffmpeg
```

Put these in one folder:

- `music_pipeline.py`
- `features_30_sec.csv` from the [GTZAN Kaggle dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) (without it, the script falls back to synthetic data and the results mean nothing)
- optionally a `my_playlist/` folder with `.wav`, `.mp3` or `.m4a` files

```bash
python music_pipeline.py                                    # classification, catalog recommenders
python music_pipeline.py --playlist-dir my_playlist --live  # plus live online suggestions
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | `features_30_sec.csv` | Feature CSV to train on |
| `--segment` | none | Segment length in seconds. Use `3` with `features_3_sec.csv` |
| `--playlist-dir` | `my_playlist` | Folder with your songs. If missing, a demo playlist from GTZAN is used |
| `--live` | off | Fetch online suggestions (needs internet and ffmpeg) |
| `--n-recs` | `10` | Number of live suggestions |

**Higher accuracy:** `python music_pipeline.py --csv features_3_sec.csv --segment 3`. This gives about ten times more training rows. Predictions are averaged over a song's segments, and the cross-validation is grouped by song, so the score stays honest.

**Jupyter users:** set the arguments in code before calling `main()`:

```python
import sys
sys.argv = ["x", "--playlist-dir", "/path/to/my_playlist", "--live"]
main()
```

and make `parse_args()` end with `return ap.parse_known_args()[0]` so Jupyter's own arguments are ignored.

---

## Outputs

Everything is written to `outputs/`.

| File | Content |
|---|---|
| `feature_table.csv` | Song-level feature table with labels |
| `1_eda.png` | Tempo, brightness, energy, harmonic/percussive balance, zero-crossing rate, MFCC heatmap per genre |
| `2_genre_models.png` | Confusion matrix and model comparison |
| `3_validation.png` | Valence/arousal density, genre centroids, feature importance |
| `4_profile_dashboard.png` | Your playlist: mood map, genre mix, comparison to catalog, track trajectory |
| `6_mood_trajectory.png` | Iso-principle path through the catalog |
| `7_live_recommendations.png` | Live suggestions on your mood map, ranked |

The terminal also prints the model table, the most confused genre pairs, statistical checks, the retrieval score, your profile and the recommendations.

---

## How it works

### Features
The 57 base features follow the GTZAN CSV: chroma, RMS, spectral centroid/bandwidth/rolloff, zero-crossing rate, harmonic and percussive components, tempo, and the mean and variance of 20 MFCCs. The 38 added ones include the harmonic-to-percussive balance, coefficients of variation, spectral ratios, and MFCC shape statistics (slope, curvature, skew, kurtosis, adjacent-coefficient spread). For your own audio files, features are computed with librosa using the same definitions.

### Classification
Each model gets its own preprocessing (robust scaling, Yeo-Johnson power transform, standardisation), fitted inside every fold so nothing leaks from the test fold. Models are ranked by cross-validated accuracy. The ensemble averages the predicted probabilities of the top four.

### Taste embedding
A linear discriminant analysis projection trained on genre labels, joined with valence and arousal, then length-normalised. Cosine similarity in this space is the song-to-song similarity. On held-out songs it finds same-genre neighbours more often than cosine similarity on raw scaled features (see Results).

### Valence and arousal
A fixed weighted combination of standardised features: loudness, brightness, tempo and percussive energy for arousal; second MFCC, chroma, harmonic share, brightness and roughness for valence. Valence is made uncorrelated with arousal, and both are squashed to the range -1 to 1.
**This is a heuristic.** It is not trained on listener ratings, so read it as "relatively calmer or more energetic than the rest of the catalog", not as an emotion label.

### Live recommendations
1. Search queries are built from your mood clusters, dominant genres and a rotating list of Indian languages.
2. Candidates come from the iTunes Search API (India store). Only tracks whose genre tag indicates an Indian language are kept.
3. Each 30-second preview is downloaded, decoded with ffmpeg and run through the same feature extraction as your songs. Previews that cannot be decoded are skipped, never filled in.
4. Candidates are scored on similarity to your tracks (mean of the three closest), fit to your mood clusters, and genre overlap, each turned into a percentile.
5. Final selection uses maximal marginal relevance (so you don't get ten near-identical songs), tops up mood clusters you listen to less, limits two tracks per artist, and drops near-copies of songs you already have.

The "match" column is the relevance percentile inside the candidate pool. It is not a probability that you will like the song.

---

## Results

A run on `features_30_sec.csv` (5-fold song-grouped cross-validation):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Soft-voting (SVM, MLP, Logistic Regression, HistGradientBoosting) | 77.9% | 0.778 |
| SVM (RBF) | 76.8% | 0.768 |
| MLP (256-128) | 74.9% | 0.747 |
| Logistic Regression | 73.2% | 0.731 |
| KNN (k=7) | 69.4% | 0.694 |

Retrieval, measured as precision@5 on held-out songs (a neighbour counts as relevant if it has the same genre): **0.535** for the taste embedding against **0.457** for raw cosine similarity.

The most confused pairs were hip-hop and reggae, and rock and disco. In the data they overlap on zero-crossing rate, tempo and loudness.

Your numbers will differ slightly by dataset version and random seed.

---

## Limitations

- **GTZAN is old and flawed.** It has repeated songs, some mislabelled tracks and distortions, and every clip is 30 seconds from the same era of recordings. Scores here say how well the model does on GTZAN, not on all music. See Sturm (2013, 2014).
- **Genre is a weak proxy for taste.** The recommender knows what a song sounds like, not who else liked it. Services like Spotify rely mostly on listening histories of millions of users (collaborative filtering), which this project does not have. See Schedl et al. (2018).
- **Previews are 30 seconds** from the middle of a track, so features describe that excerpt, not the whole song.
- **Language detection is based on iTunes genre tags**, which are patchy. Expect fewer results than a language-aware catalog would give.
- **The demo playlist** (when you have no `my_playlist/` folder) is made of songs the model was trained on, so its genre probabilities are optimistic.
- **The live part needs internet and ffmpeg.** The iTunes endpoint is public and unauthenticated; it can rate-limit or change.

---

## Customising

- **Other languages:** in `main()`, create the recommender with `LiveRecommender(bundle, languages=["Hindi", "English", "Korean"], indian_only=False, segment_sec=args.segment)`.
- **Repeatable suggestions:** pass `seed=1` to `.recommend(...)`.
- **Diversity vs relevance:** the `lam` argument of `rank_candidates` (default 0.7). Lower gives more variety.
- **Better valence/arousal:** replace `AffectModel` with a regressor trained on a labelled set such as DEAM (Aljanaki et al., 2017).

---

## Project layout

```
music_pipeline.py     everything: features, models, profile, recommenders
features_30_sec.csv   GTZAN features (you provide)
my_playlist/          your audio files (optional)
outputs/              figures and the feature table
```

Inside `music_pipeline.py`, the main pieces are `engineer_features`, `benchmark_genre_models`, `GenreEnsemble`, `AffectModel`, `TasteEmbedder`, `evaluate_retrieval`, `build_profile`, `similar_songs`, `mood_trajectory`, `rank_candidates` and `LiveRecommender`.

---

## References

**Dataset and audio features**
- Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing*, 10(5), 293-302.
- Sturm, B. L. (2013). The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use. arXiv:1306.1461.
- Sturm, B. L. (2014). The state of the art ten years after a state of the art: Future research in music information retrieval. *Journal of New Music Research*, 43(2), 147-172.
- McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference*.
- Olteanu, A. GTZAN Dataset - Music Genre Classification (Kaggle).

**Methods**
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- Yeo, I.-K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or symmetry. *Biometrika*, 87(4), 954-959.
- Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *Proceedings of SIGIR '98*.
- Steck, H. (2018). Calibrated recommendations. *Proceedings of the 12th ACM Conference on Recommender Systems (RecSys '18)*.
- Schedl, M., et al. (2018). Current challenges and visions in music recommender systems research. *International Journal of Multimedia Information Retrieval*, 7, 95-116.

**Emotion and mood**
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.
- Aljanaki, A., Yang, Y.-H., & Soleymani, M. (2017). Developing a benchmark for emotional analysis of music. *PLOS ONE*, 12(3), e0173392.
- Heiderscheit, A., & Madson, A. (2015). Use of the iso principle as a central method in mood management: A music psychotherapy clinical case study. *Music Therapy Perspectives*, 33(1), 45-52.

---

