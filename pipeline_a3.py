# -*- coding: utf-8 -*-
"""
Pipeline A3 - Projeto Aplicado II (Mackenzie)
Steam Reviews + generos (base gerada pelo merge_generos.py)

O que este script faz, na ordem em que a Etapa 2 prometeu:
  1. Le o CSV grande em chunks e coleta: volume, nulos e valores de review_score/review_votes
     na base INTEIRA, e separa uma amostra aleatoria (por padrao 5%) para modelar.
  2. Tratamento do texto: minusculas, remocao de links e pontuacao, stopwords
     (preservando negacoes), lematizacao.
  3. EDA: balanceamento das classes, tamanho dos textos, taxa de aprovacao por genero.
  4. Particao treino/teste (80/20 estratificada) e TF-IDF ajustado SO no treino.
  5. Validacao cruzada K-Fold (k=5) com Pipeline (TF-IDF refeito em cada fold, sem vazamento).
  6. Treino final e avaliacao no teste: Regressao Logistica, Naive Bayes, Random Forest.
     Metricas: acuracia, precisao, recall, F1, F1 macro, ROC-AUC, matriz de confusao.
  7. Acuracia por genero (melhor modelo) e palavras mais associadas a cada classe.
  8. Salva tudo na pasta de saida: resultados.txt, CSVs e graficos PNG.

Uso (no terminal do PyCharm, com o ambiente virtual):
  python -m pip install pandas scikit-learn matplotlib nltk
  python pipeline_a3.py
  python pipeline_a3.py --entrada "D:\\pasta\\reviews_com_genero.csv" --saida "D:\\pasta\\resultados_A3"

Opcoes uteis:
  --frac 0.02      usa 2% da base (mais rapido, para testar)
  --sem_rf         pula a Random Forest (a parte mais lenta)
"""

import argparse
import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Negacoes invertem o sentido ("not good"), entao NAO entram na lista de stopwords.
NEGACOES = {"not", "no", "nor", "never", "none", "nobody", "nothing", "nowhere",
            "cannot", "neither", "without"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGACOES

RELATORIO = []


def ptbr(texto):
    """Converte numeros do formato americano (6,417,106 / 0.879) para o brasileiro (6.417.106 / 0,879)."""
    def troca(m):
        return m.group(0).replace(",", "\u00a7").replace(".", ",").replace("\u00a7", ".")
    return re.sub(r"\d+(?:[.,]\d+)*", troca, texto)


def br(v, casas=0):
    """Numero no formato brasileiro (milhar com ponto, decimal com virgula) para os graficos."""
    return f"{v:,.{casas}f}".replace(",", "\u00a7").replace(".", ",").replace("\u00a7", ".")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def R(texto=""):
    """Imprime e guarda a linha no relatorio de resultados."""
    print(texto, flush=True)
    RELATORIO.append(str(texto))


# ----------------------------------------------------------------------------
# 1. Leitura em chunks
# ----------------------------------------------------------------------------
def ler_e_amostrar(caminho, frac, chunk, seed):
    total = 0
    nulos = None
    cont_score = pd.Series(dtype="float64")
    cont_votes = pd.Series(dtype="float64")
    amostras = []
    colunas = None

    for i, ch in enumerate(pd.read_csv(caminho, chunksize=chunk), start=1):
        if colunas is None:
            colunas = list(ch.columns)
        total += len(ch)
        n = ch.isna().sum()
        nulos = n if nulos is None else nulos.add(n, fill_value=0)
        cont_score = cont_score.add(ch["review_score"].value_counts(), fill_value=0)
        cont_votes = cont_votes.add(ch["review_votes"].value_counts(), fill_value=0)
        amostras.append(ch.sample(frac=frac, random_state=seed + i))
        log(f"  chunk {i}: {total:,} linhas lidas")

    amostra = pd.concat(amostras, ignore_index=True)
    return (amostra, total, nulos.astype(int), cont_score.astype(int).sort_index(),
            cont_votes.astype(int).sort_index(), colunas)


# ----------------------------------------------------------------------------
# 2. Tratamento de texto
# ----------------------------------------------------------------------------
def normalizar(serie):
    s = serie.astype(str).str.lower()
    s = s.str.replace("\u2019", "'", regex=False)
    s = s.str.replace(r"http\S+|www\.\S+", " ", regex=True)      # links
    s = s.str.replace(r"\bcan't\b", "can not", regex=True)
    s = s.str.replace(r"\bwon't\b", "will not", regex=True)
    s = s.str.replace(r"n't\b", " not", regex=True)              # don't -> do not
    s = s.str.replace("'", "", regex=False)
    s = s.str.replace(r"[^a-z0-9\s]", " ", regex=True)           # pontuacao/especiais
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def criar_lematizador():
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        lem = WordNetLemmatizer()
        lem.lemmatize("tests")  # forca o carregamento; falha aqui se o corpus nao existir
        return lem
    except Exception as e:  # noqa: BLE001
        log(f"AVISO: lematizacao indisponivel ({e}). Seguindo SEM lematizar.")
        return None


def remover_stopwords_e_lematizar(textos, lem):
    cache = {}

    def lema(w):
        if lem is None:
            return w
        if w not in cache:
            cache[w] = lem.lemmatize(lem.lemmatize(w, pos="v"), pos="n")
        return cache[w]

    saida = []
    for t in textos:
        saida.append(" ".join(lema(w) for w in t.split() if w not in STOPWORDS and len(w) > 1))
    return saida


def novo_vetorizador(max_features):
    return TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=5,
                           max_features=max_features, sublinear_tf=True)


def criar_modelos(seed, com_rf):
    modelos = {
        "Regressão Logística": LogisticRegression(C=2.0, solver="liblinear",
                                                  class_weight="balanced", max_iter=200),
        "Naive Bayes (Multinomial)": MultinomialNB(alpha=0.5),
    }
    if com_rf:
        modelos["Random Forest"] = RandomForestClassifier(
            n_estimators=100, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    return modelos


# ----------------------------------------------------------------------------
# Avaliacao
# ----------------------------------------------------------------------------
def avaliar(nome, modelo, Xte, yte, tempo):
    pred = modelo.predict(Xte)
    proba = modelo.predict_proba(Xte)[:, 1]
    linha = {
        "Modelo": nome,
        "Acurácia": accuracy_score(yte, pred),
        "Precisão (positiva)": precision_score(yte, pred),
        "Recall (positiva)": recall_score(yte, pred),
        "Recall (negativa)": recall_score(yte, pred, pos_label=0),
        "F1 (positiva)": f1_score(yte, pred),
        "F1 macro": f1_score(yte, pred, average="macro"),
        "ROC-AUC": roc_auc_score(yte, proba),
        "Tempo de treino (s)": round(tempo, 1),
    }
    return linha, pred


def validacao_cruzada(nome, modelo, textos, y, max_features, seed, k):
    pipe = Pipeline([("tfidf", novo_vetorizador(max_features)), ("clf", modelo)])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scoring = {"acuracia": "accuracy", "precisao": "precision", "recall": "recall",
               "f1": "f1", "f1_macro": "f1_macro"}
    res = cross_validate(pipe, textos, y, cv=skf, scoring=scoring, n_jobs=1)
    linha = {"Modelo": nome, "Amostra CV": len(y), "Folds": k}
    for chave in scoring:
        linha[f"{chave} (média)"] = res[f"test_{chave}"].mean()
        linha[f"{chave} (desvio)"] = res[f"test_{chave}"].std()
    return linha


# ----------------------------------------------------------------------------
# Graficos
# ----------------------------------------------------------------------------
def gravar(escrever, caminho):
    """Roda escrever(caminho). Se o Windows recusar (arquivo aberto em outro programa, pasta
    sincronizada etc.), grava com o prefixo da hora em vez de derrubar o script inteiro."""
    try:
        escrever(caminho)
        return caminho
    except OSError as e:
        alt = os.path.join(os.path.dirname(caminho),
                           time.strftime("%H%M%S_") + os.path.basename(caminho))
        log(f"AVISO: nao deu para gravar {os.path.basename(caminho)} ({e}). "
            f"Tentando {os.path.basename(alt)}...")
        try:
            escrever(alt)
            return alt
        except OSError as e2:
            log(f"AVISO: {os.path.basename(caminho)} NAO foi salvo ({e2}). "
                "Feche programas que estejam usando a pasta (visualizador de imagens, Excel) "
                "ou use --saida com outra pasta.")
            return None


def salvar(fig, pasta, nome):
    fig.tight_layout()
    destino = gravar(lambda c: fig.savefig(c, dpi=150, bbox_inches="tight"),
                     os.path.join(pasta, nome))
    plt.close(fig)
    if destino:
        log(f"  grafico salvo: {os.path.basename(destino)}")


def graf_classes(df, pasta):
    contagem = df["y"].value_counts().sort_index()
    rot = ["Não recomenda", "Recomenda"]
    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.bar(rot, [contagem.get(0, 0), contagem.get(1, 0)], color=["#c0392b", "#27ae60"])
    for b in barras:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), br(b.get_height()),
                ha="center", va="bottom")
    ax.set_title("Distribuição das classes (amostra tratada)")
    ax.set_ylabel("Reviews")
    salvar(fig, pasta, "01_distribuicao_classes.png")


def graf_tamanho(df, pasta):
    limite = df["n_palavras"].quantile(0.99)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.loc[df["n_palavras"] <= limite, "n_palavras"], bins=50, color="#2980b9")
    ax.set_title("Tamanho das reviews após tratamento (palavras, até o percentil 99)")
    ax.set_xlabel("Palavras por review")
    ax.set_ylabel("Frequência")
    salvar(fig, pasta, "02_tamanho_textos.png")


def graf_matrizes(resultados_cm, pasta):
    n = len(resultados_cm)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2))
    axes = np.atleast_1d(axes)
    for ax, (nome, cm) in zip(axes, resultados_cm.items()):
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{br(cm[i, j])}\n({norm[i, j]:.0%})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["Prev. não", "Prev. sim"])
        ax.set_yticks([0, 1], ["Real não", "Real sim"])
        ax.set_title(nome)
    salvar(fig, pasta, "03_matrizes_confusao.png")


def graf_modelos(tab, pasta):
    metricas = ["Acurácia", "F1 (positiva)", "F1 macro", "ROC-AUC"]
    x = np.arange(len(tab))
    largura = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, m in enumerate(metricas):
        ax.bar(x + k * largura, tab[m], largura, label=m)
    ax.set_xticks(x + 1.5 * largura, tab["Modelo"])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 1)))
    ax.set_title("Comparação dos modelos (conjunto de teste)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    salvar(fig, pasta, "04_comparacao_modelos.png")


def graf_generos(tab_gen, pasta, min_n, media):
    t = tab_gen[tab_gen["reviews"] >= min_n].sort_values("pct_positivas")
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(t))))
    ax.barh(t.index, t["pct_positivas"] * 100, color="#16a085")
    for y_, v in enumerate(t["pct_positivas"] * 100):
        ax.text(v, y_, f" {br(v, 1)}%", va="center")
    ax.axvline(media * 100, color="gray", linestyle="--", label=f"Média geral: {br(media * 100, 1)}%")
    ax.set_xlabel("% de reviews que recomendam")
    ax.set_title(f"Aceitação por gênero (gêneros com ao menos {br(min_n)} reviews)")
    ax.set_xlim(0, 105)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    salvar(fig, pasta, "05_generos_taxa_positiva.png")


def graf_palavras(top_pos, top_neg, pasta):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].barh(top_pos["termo"][::-1], top_pos["peso"][::-1], color="#27ae60")
    axes[0].set_title("Termos mais associados a RECOMENDAR")
    axes[1].barh(top_neg["termo"][::-1], -top_neg["peso"][::-1], color="#c0392b")
    axes[1].set_title("Termos mais associados a NÃO recomendar")
    for ax in axes:
        ax.set_xlabel("Peso (coeficiente da Regressão Logística)")
    salvar(fig, pasta, "06_palavras_top.png")


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default=r"C:\Users\davim\Downloads\reviews_com_genero.csv")
    ap.add_argument("--saida", default=r"C:\Users\davim\Downloads\resultados_A3")
    ap.add_argument("--frac", type=float, default=0.05, help="fracao da base usada na modelagem")
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_features", type=int, default=50_000)
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--cv_n", type=int, default=60_000, help="linhas usadas na validacao cruzada")
    ap.add_argument("--cv_rf_n", type=int, default=20_000, help="linhas da validacao cruzada da RF")
    ap.add_argument("--rf_n", type=int, default=80_000, help="linhas de treino da Random Forest")
    ap.add_argument("--min_genero", type=int, default=1000)
    ap.add_argument("--sem_rf", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.saida, exist_ok=True)
    inicio = time.time()

    # ---------------- 1. Origem e volume ----------------
    log("Etapa 1/7: lendo a base em chunks...")
    amostra, total, nulos, cont_score, cont_votes, colunas = ler_e_amostrar(
        args.entrada, args.frac, args.chunk, args.seed)

    R("=" * 70)
    R("RESULTADOS PRELIMINARES - PROJETO APLICADO II (A3)")
    R("=" * 70)
    R("")
    R("1. ORIGEM E VOLUME DOS DADOS")
    R(f"Arquivo analisado: {os.path.basename(args.entrada)} (base de reviews da Steam enriquecida com genero)")
    R(f"Total de registros na base completa: {total:,}")
    R(f"Colunas ({len(colunas)}): {', '.join(colunas)}")
    R("")
    R("Valores nulos por coluna (base completa):")
    for c in colunas:
        R(f"  {c}: {nulos.get(c, 0):,} ({nulos.get(c, 0) / total:.2%})")
    R("")
    R("Valores encontrados em review_score (base completa):")
    for v, n in cont_score.items():
        R(f"  {v}: {n:,} ({n / total:.2%})")
    R("Valores encontrados em review_votes (base completa):")
    for v, n in cont_votes.items():
        R(f"  {v}: {n:,} ({n / total:.2%})")
    R("(Use estes valores para conferir o dicionario de dados do relatorio.)")
    R("")

    # ---------------- 2. Tratamento ----------------
    log("Etapa 2/7: tratamento do texto...")
    n0 = len(amostra)
    df = amostra.dropna(subset=["review_text", "review_score"]).copy()
    removidos_nulos = n0 - len(df)

    df["texto_norm"] = normalizar(df["review_text"])
    lem = criar_lematizador()
    df["texto"] = remover_stopwords_e_lematizar(df["texto_norm"].tolist(), lem)

    antes = len(df)
    df = df[df["texto"].str.len() >= 2].copy()
    removidos_vazios = antes - len(df)

    antes = len(df)
    df = df.drop_duplicates(subset=["texto", "review_score"]).copy()
    removidos_dup = antes - len(df)

    df["y"] = (df["review_score"] > 0).astype(int)
    df["n_palavras"] = df["texto"].str.split().str.len()
    tem_genero = "genres" in df.columns
    df = df.reset_index(drop=True)

    R("2. TRATAMENTO DA BASE")
    R(f"Amostra aleatoria usada na modelagem: {n0:,} reviews ({args.frac:.0%} da base)")
    R(f"  - removidos por texto/rotulo nulo: {removidos_nulos:,}")
    R(f"  - removidos por texto vazio apos limpeza: {removidos_vazios:,}")
    R(f"  - removidos por duplicidade (mesmo texto e mesmo rotulo): {removidos_dup:,}")
    R(f"  - base final tratada: {len(df):,} reviews")
    R("Etapas aplicadas: minusculas; remocao de links, pontuacao e caracteres especiais; "
      "stopwords em ingles (negacoes preservadas); "
      + ("lematizacao (WordNet); " if lem is not None else "SEM lematizacao (nltk indisponivel); ")
      + f"TF-IDF (unigramas e bigramas, ate {args.max_features:,} termos, min_df=5).")
    R("")

    # ---------------- 3. EDA ----------------
    log("Etapa 3/7: analise exploratoria...")
    pos = df["y"].mean()
    baseline = max(pos, 1 - pos)
    R("3. ANALISE EXPLORATORIA (amostra tratada)")
    R(f"Reviews que recomendam: {pos:.2%} | nao recomendam: {1 - pos:.2%}")
    R(f"Baseline (prever sempre a classe majoritaria): acuracia de {baseline:.2%}")
    R("  -> qualquer modelo precisa superar esse valor; por isso o F1 macro e o recall da classe "
      "negativa tambem sao reportados.")
    R(f"Tamanho das reviews apos tratamento (palavras): media {df['n_palavras'].mean():.1f}, "
      f"mediana {df['n_palavras'].median():.0f}, percentil 95 = {df['n_palavras'].quantile(0.95):.0f}, "
      f"maximo {df['n_palavras'].max():.0f}")
    R(f"Jogos distintos na amostra: {df['app_id'].nunique():,}")

    graf_classes(df, args.saida)
    graf_tamanho(df, args.saida)

    tab_gen = None
    media_genero = pos
    if tem_genero:
        com_gen = df.dropna(subset=["genres"])
        media_genero = com_gen["y"].mean()
        g = com_gen[["genres", "y"]].copy()
        g["genero"] = g["genres"].str.split(";")
        g = g.explode("genero")
        g["genero"] = g["genero"].str.strip()
        tab_gen = g.groupby("genero").agg(reviews=("y", "size"), pct_positivas=("y", "mean"))
        tab_gen = tab_gen.sort_values("reviews", ascending=False)
        R(f"Reviews com genero identificado na amostra tratada: {len(com_gen):,} ({len(com_gen) / len(df):.1%})")
        R("Obs.: um jogo pode ter varios generos; a review conta em TODOS os generos do jogo "
          "(por isso a soma por genero ultrapassa o total de reviews).")
    R("")

    # ---------------- 4. Particao ----------------
    log("Etapa 4/7: particao treino/teste e TF-IDF...")
    idx_tr, idx_te = train_test_split(df.index.values, test_size=0.2, random_state=args.seed,
                                      stratify=df["y"])
    vec = novo_vetorizador(args.max_features)
    Xtr = vec.fit_transform(df.loc[idx_tr, "texto"])
    Xte = vec.transform(df.loc[idx_te, "texto"])
    ytr, yte = df.loc[idx_tr, "y"].values, df.loc[idx_te, "y"].values
    R("4. PARTICAO DOS DADOS")
    R(f"Treino: {len(idx_tr):,} reviews (80%) | Teste: {len(idx_te):,} reviews (20%), "
      "estratificado pela classe, seed fixa.")
    R(f"Vocabulario TF-IDF (ajustado somente no treino): {Xtr.shape[1]:,} termos")
    R("")

    # ---------------- 5. Validacao cruzada ----------------
    log("Etapa 5/7: validacao cruzada K-Fold (esta etapa demora)...")
    rng = np.random.default_rng(args.seed)
    cv_idx = rng.choice(idx_tr, size=min(args.cv_n, len(idx_tr)), replace=False)
    cv_rf_idx = rng.choice(idx_tr, size=min(args.cv_rf_n, len(idx_tr)), replace=False)
    linhas_cv = []
    for nome, modelo in criar_modelos(args.seed, not args.sem_rf).items():
        idx = cv_rf_idx if nome == "Random Forest" else cv_idx
        t0 = time.time()
        linhas_cv.append(validacao_cruzada(nome, modelo, df.loc[idx, "texto"].values,
                                           df.loc[idx, "y"].values, args.max_features,
                                           args.seed, args.kfolds))
        log(f"  {nome}: CV concluida em {time.time() - t0:.0f}s")
    tab_cv = pd.DataFrame(linhas_cv)
    gravar(lambda c: tab_cv.to_csv(c, index=False, encoding="utf-8-sig"),
           os.path.join(args.saida, "metricas_validacao_cruzada.csv"))
    R(f"5. VALIDACAO CRUZADA (K-Fold, k={args.kfolds}; TF-IDF refeito em cada fold)")
    for _, r in tab_cv.iterrows():
        R(f"{r['Modelo']} (amostra de {int(r['Amostra CV']):,} reviews): "
          f"acuracia {r['acuracia (média)']:.3f} +/- {r['acuracia (desvio)']:.3f} | "
          f"precisao {r['precisao (média)']:.3f} | recall {r['recall (média)']:.3f} | "
          f"F1 {r['f1 (média)']:.3f} +/- {r['f1 (desvio)']:.3f} | F1 macro {r['f1_macro (média)']:.3f}")
    R("")

    # ---------------- 6. Treino final e teste ----------------
    log("Etapa 6/7: treino final e avaliacao no teste...")
    linhas, matrizes, preds = [], {}, {}
    modelos = criar_modelos(args.seed, not args.sem_rf)
    for nome, modelo in modelos.items():
        t0 = time.time()
        if nome == "Random Forest" and Xtr.shape[0] > args.rf_n:
            sub = rng.choice(Xtr.shape[0], size=args.rf_n, replace=False)
            modelo.fit(Xtr[sub], ytr[sub])
        else:
            modelo.fit(Xtr, ytr)
        linha, pred = avaliar(nome, modelo, Xte, yte, time.time() - t0)
        linhas.append(linha)
        preds[nome] = pred
        matrizes[nome] = confusion_matrix(yte, pred)
        log(f"  {nome}: acuracia {linha['Acurácia']:.3f} | F1 macro {linha['F1 macro']:.3f}")
    tab = pd.DataFrame(linhas)
    gravar(lambda c: tab.to_csv(c, index=False, encoding="utf-8-sig"),
           os.path.join(args.saida, "metricas_teste.csv"))
    graf_matrizes(matrizes, args.saida)
    graf_modelos(tab, args.saida)

    melhor = tab.sort_values("F1 macro", ascending=False).iloc[0]["Modelo"]
    R("6. AVALIACAO NO CONJUNTO DE TESTE")
    R(f"Baseline de acuracia (classe majoritaria): {baseline:.3f}")
    for _, r in tab.iterrows():
        R(f"{r['Modelo']}: acuracia {r['Acurácia']:.3f} | precisao {r['Precisão (positiva)']:.3f} | "
          f"recall (positiva) {r['Recall (positiva)']:.3f} | recall (negativa) {r['Recall (negativa)']:.3f} | "
          f"F1 {r['F1 (positiva)']:.3f} | F1 macro {r['F1 macro']:.3f} | ROC-AUC {r['ROC-AUC']:.3f} | "
          f"treino {r['Tempo de treino (s)']}s")
        cm = matrizes[r["Modelo"]]
        R(f"   matriz de confusao [real nao: prev nao / prev sim ; real sim: prev nao / prev sim] = "
          f"[{cm[0, 0]:,} / {cm[0, 1]:,} ; {cm[1, 0]:,} / {cm[1, 1]:,}]")
    R(f"Melhor modelo por F1 macro: {melhor}")
    R("Criterio de escolha: F1 macro (media do F1 das duas classes). O F1 da classe positiva fica alto "
      "ate para modelos que erram muito a classe negativa, por causa do desbalanceamento.")
    atingiram = [r["Modelo"] for _, r in tab.iterrows() if r["F1 macro"] > 0.75]
    R(f"Meta da Etapa 2 (F1 > 0.75, medida pelo F1 macro): atingida por {len(atingiram)} de {len(tab)} "
      f"modelos ({', '.join(atingiram) if atingiram else 'nenhum'}).")
    if "Random Forest" in modelos and Xtr.shape[0] > args.rf_n:
        R(f"(Random Forest treinada com subamostra de {args.rf_n:,} reviews do treino, por custo computacional.)")
    R("Obs.: Regressao Logistica e Random Forest usam class_weight balanceado; Naive Bayes usa as "
      "probabilidades a priori da propria base.")
    R("")

    # ---------------- 7. Generos e palavras ----------------
    log("Etapa 7/7: genero e palavras mais importantes...")
    R("7. ACEITACAO E DESEMPENHO POR GENERO")
    if tem_genero and tab_gen is not None:
        te = df.loc[idx_te, ["genres", "y"]].copy()
        te["acerto"] = (preds[melhor] == yte).astype(int)
        te = te.dropna(subset=["genres"])
        te["genero"] = te["genres"].str.split(";")
        te = te.explode("genero")
        te["genero"] = te["genero"].str.strip()
        acc_gen = te.groupby("genero").agg(reviews_teste=("acerto", "size"),
                                           acuracia_modelo=("acerto", "mean"))
        tab_gen = tab_gen.join(acc_gen)
        gravar(lambda c: tab_gen.to_csv(c, encoding="utf-8-sig"),
               os.path.join(args.saida, "generos.csv"))
        graf_generos(tab_gen, args.saida, args.min_genero, media_genero)
        grandes = tab_gen[tab_gen["reviews"] >= args.min_genero].sort_values("pct_positivas",
                                                                              ascending=False)
        R(f"Media geral de aprovacao (reviews com genero): {media_genero:.1%}")
        R(f"Generos com ao menos {args.min_genero:,} reviews na amostra "
          f"(% que recomenda | acuracia do melhor modelo no teste):")
        for gen, r in grandes.iterrows():
            acc = f"{r['acuracia_modelo']:.3f}" if pd.notna(r["acuracia_modelo"]) else "n/d"
            R(f"  {gen}: {r['pct_positivas']:.1%} | {acc} | {int(r['reviews']):,} reviews")
    else:
        R("Coluna de genero nao encontrada na base.")
    R("")

    lr = modelos["Regressão Logística"]
    termos = np.array(vec.get_feature_names_out())
    ordem = np.argsort(lr.coef_[0])
    top_neg = pd.DataFrame({"termo": termos[ordem[:15]], "peso": lr.coef_[0][ordem[:15]]})
    top_pos = pd.DataFrame({"termo": termos[ordem[::-1][:15]], "peso": lr.coef_[0][ordem[::-1][:15]]})
    termos_df = pd.concat([top_pos.assign(classe="recomenda"), top_neg.assign(classe="nao recomenda")])
    gravar(lambda c: termos_df.to_csv(c, index=False, encoding="utf-8-sig"),
           os.path.join(args.saida, "termos_mais_importantes.csv"))
    graf_palavras(top_pos, top_neg, args.saida)
    R("8. TERMOS MAIS ASSOCIADOS A CADA CLASSE (Regressao Logistica)")
    R("Recomenda: " + ", ".join(top_pos["termo"]))
    R("Nao recomenda: " + ", ".join(top_neg["termo"]))
    R("")
    R(f"Tempo total de execucao: {(time.time() - inicio) / 60:.1f} minutos")

    texto_final = ptbr("\n".join(RELATORIO))

    def _escrever_txt(c):
        with open(c, "w", encoding="utf-8") as f:
            f.write(texto_final)

    gravar(_escrever_txt, os.path.join(args.saida, "resultados.txt"))
    log(f"Pronto! Arquivos salvos em: {args.saida}")


if __name__ == "__main__":
    main()
