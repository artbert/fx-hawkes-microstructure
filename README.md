# High-Frequency FX Microstructure Analysis: Hawkes Processes on EURUSD

## 📌 Executive Summary

Projekt koncentruje się na modelowaniu dynamiki mikrostruktury rynku walutowego (EURUSD) przy użyciu **samowzbudzających się procesów punktowych (Hawkes Processes)**. Analiza przechodzi od surowej statystyki danych tickowych, przez dekompozycję sezonowości intraday, aż po zaawansowane modelowanie interakcji Bid-Ask w wymiarze 2D.

Kluczowym osiągnięciem projektu jest identyfikacja i rozwiązanie problemów zbieżności modelu wynikających z ograniczeń technicznych danych (rozdzielczość 1ms), co pozwoliło na uzyskanie stabilnych i ekonomicznie uzasadnionych parametrów endogeniczności rynku.

## 🚀 Key Technical Highlights

* **De-biasing Endogeneity:** Udowodniono, że ignorowanie sezonowości intraday zawyża współczynnik samowzbudzania ($\alpha$) z realnych **50% do aż 89%**.
* **Numerical Stability & Constraints:** Rozwiązano problem degeneracji modelu 2D ($\beta \to \infty$) poprzez zastosowanie optymalizacji z restrykcjami, dostosowując model do fizycznych limitów infrastruktury rynkowej.
* **Cross-Excitation Discovery:** Wykazano, że dynamika EURUSD jest napędzana głównie przez **interakcje krzyżowe (~0.65)**, a nie samowzbudzanie wewnątrz jednej strony arkusza (~0.07).
* **High-Performance Estimation:** Implementacja estymacji Maximum Likelihood (MLE) z wykorzystaniem rekurencyjnej formy jądra wykładniczego, co zapewnia liniową złożoność obliczeniową.

## 🛠 Tech Stack

* **Language:** Python 3.x
* **Data Science:** Pandas, NumPy, SciPy (Optimization)
* **Visualization:** Matplotlib, Seaborn
* **Storage:** Apache Parquet (high-efficiency I/O)

## 🚀 How to Run

### Option 1: Google Colab (Recommended)

Najszybszy sposób na przetestowanie analizy. Kliknij poniższy badge, a następnie uruchom pierwszą komórkę w notebooku, aby automatycznie skonfigurować środowisko:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/artbert/fx-hawkes-microstructure/blob/main/notebook.ipynb)

### Option 2: Local Installation

Jeśli wolisz pracować lokalnie, upewnij się, że masz zainstalowanego Pythona 3.8+:

1. **Sklonuj repozytorium:**
```bash
git clone https://github.com/artbert/fx-hawkes-microstructure.git
cd fx-hawkes-microstructure

```


2. **Zainstaluj zależności:**
```bash
pip install -r requirements.txt

```


3. **Uruchom Jupyter Lab/Notebook:**
```bash
jupyter notebook notebook.ipynb

```

## 📈 Selected Results

| Model Configuration | Endogeneity ($\alpha$) | Market Memory ($1/\beta$) | AIC Improvement |
| --- | --- | --- | --- |
| Baseline (Constant) | ~84% | ~3.0s | Baseline |
| **Seasonal (pconst)** | **~55%** | **~0.8s** | **Significant Drop** |

## 📂 Repository Structure

* `notebook.ipynb`: Kompletny workflow analityczny z opisami merytorycznymi.
* `src/utils.py`: Silnik obliczeniowy (estymacja MLE, przetwarzanie danych, wizualizacje).
* `data/`: Przykładowe próbki danych oraz wyczyszczone pliki Parquet.