# Pessoa A – Interface e Explicabilidade

Python 3 · Tkinter + matplotlib · nibabel · PyTorch (só p/ inferência e Grad-CAM)

## Passo a passo

### Passo 1: Preparar a pasta e o ambiente

Coloque todos os arquivos `pa_*.py`, `make_synthetic_data.py`, `test_pa_headless.py` e `test_pa_torch.py` **na mesma pasta**. Abra um terminal nela.

```bash
python -m venv venv
```

Ativar: no Windows `venv\Scripts\activate`; no Linux/macOS `source venv/bin/activate`.

```bash
pip install numpy scipy nibabel pandas matplotlib scikit-learn
pip install torch torchvision
```

Se der problema com o torch, siga o comando de instalação em pytorch.org para o seu sistema.

### Passo 2: Verificar o Tkinter

```bash
python -c "import tkinter; print('tk ok')"
```

No Ubuntu/Debian, se falhar: `sudo apt install python3-tk`.

### Passo 3: Teste sem torch

```bash
python test_pa_headless.py
```

Esperado: 9 linhas `OK` e a frase “TODOS OS TESTES SEM TORCH PASSARAM”.

### Passo 4: Teste com torch

```bash
python test_pa_torch.py
```

Esperado: 5 linhas `OK` e “TODOS OS TESTES COM TORCH PASSARAM”. Este é o primeiro teste real do Grad-CAM. Se algum falhar, me mande a mensagem de erro completa.

### Passo 5: Gerar dados sintéticos

```bash
python make_synthetic_data.py ./sintetico
```

Cria a pasta `sintetico` com 16 pacientes.

### Passo 6: Abrir a interface

```bash
python pa_gui_app.py
```

Clique em **“Modelo DEMO (teste da interface)”** no topo. Ele cria dois modelos de brinquedo com pesos aleatórios, então as probabilidades não têm significado.

## Módulos
| Arquivo | Função | Item do enunciado |
|---|---|---|
| `pa_config.py` | **Único lugar a ajustar** após baixar o dataset (padrões de nome de arquivo, colunas do CSV, orientação) | – |
| `pa_nii_io.py` | Leitura `.nii`, normalização, indexação paciente→fases+máscara+rótulo+split | a |
| `pa_augment.py` | 10 variações (5 rotações × orig/espelhada), análise e plano de balanceamento | b |
| `pa_slices.py` | central, maior área, média, máximo, média ponderada pela área tumoral | c |
| `pa_models.py` | **Contrato com B e C** (checkpoint, `ModelSpec`, pré-processamento, modelo DEMO) | d/e |
| `pa_gradcam.py` | Grad-CAM com hooks, métricas calor×tumor, figuras de acertos/erros | e |
| `pa_metrics.py`, `pa_folder_eval.py` | métricas e “classificar pasta nova” (sem retreinar) | relatório / teste surpresa |
| `pa_gui_*.py`, `pa_gui_app.py` | abas e janela principal (`python pa_gui_app.py`) | todos |
| `make_synthetic_data.py`, `test_pa_headless.py`, `test_pa_torch.py` | dados sintéticos e testes | – |

## O que B e C precisam entregar
1. Salvar o checkpoint no formato descrito no topo de `pa_models.py`
   (`arch`, `state_dict`, `mode`, `in_channels`, `input_size`, `output_kind`, `imagenet_norm`, `threshold`).
2. Registrar a arquitetura: `pa_models.register_arch("cnn", builder)` (idem `resnet50`, `three_branch`).
3. **Pré-processamento idêntico ao do treino**: `pa_models.build_input` usa normalização por percentil
   (1–99) por volume/fase, canais na ordem pre, early, late. Se B usar outra, alterar só `build_input`.
4. Grad-CAM: expor `model.gradcam_layers` (lista de módulos) nos modelos de ramos; ResNet50 usa `layer4`
   e a CNN própria a última `Conv2d` automaticamente.
5. (opcional) JSON por experimento para a aba *Treino/Resultados* (formato em `pa_gui_results.py`).
6. `pa_gui_app._import_team_architectures` importa `pb_models`/`pc_models` se existirem – ajustar os nomes.

## Pontos a validar quando o dataset real estiver em mãos
- `PHASE_PATTERNS`, `MASK_PATTERNS`, colunas do CSV: **foram escritos sem ver o dataset** (suposição: estilo nnU-Net
  `_0000/_0001/_0002` e `*seg*`). O resumo da indexação aparece na aba Visualizar.
- Orientação: comparar com o MRIcro e ajustar o combo “Orientação”/“Espelhar E/D” (e `DEFAULT_*` em `pa_config.py`).
- Normalização por percentil por volume remove o contraste absoluto entre fases; B deve decidir se mantém.

## Fusão em arquivo único (exigência do enunciado)
Os módulos usam só imports de topo (`from pa_x import ...`), sem dependências circulares; na integração,
concatenar na ordem: config → nii_io → augment → metrics → models → slices → gradcam → folder_eval → gui_common →
abas → app, removendo as linhas `from pa_* import` e mantendo o cabeçalho com nomes/matrículas.

## Estrutura final sugerida

```text
projeto-pcr/
├── README.md                  # como rodar + quem é dono de quê
├── requirements.txt
├── .gitignore
├── docs/
│   ├── CONTRATO.md            # checkpoint, pré-processamento, Grad-CAM (copiar do topo de pa_models.py)
│   ├── DECLARACAO_IAG.md      # preencher conforme cada um usa IA (obrigatório)
│   └── CHECKLIST.md
├── src/                       # tudo plano, rodar sempre de dentro daqui
│   ├── pa_*.py               # Pessoa A (os 16 módulos atuais)
│   ├── make_synthetic_data.py
│   ├── test_pa_headless.py
│   ├── test_pa_torch.py
│   ├── pb_*.py               # Pessoa B (ex.: pb_models.py, pb_train.py)
│   ├── pc_*.py               # Pessoa C (ex.: pc_models.py, pc_train.py)
│   └── test_pb_*.py / test_pc_*.py
├── scripts/
│   └── build_single_file.py   # junta tudo em arquivo único (ainda não existe)
├── results/                   # JSONs de loss e métricas (pequenos, entram no git)
├── report/                    # LaTeX SBC + figuras finais
├── checkpoints/               # pesos (.pt), NÃO entram no git
└── data/                      # dataset, NÃO entra no git
```
