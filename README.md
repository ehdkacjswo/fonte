# **Fonte: Finding Bug Inducing Commit From Failure (ICSE'23)**
[![DOI](https://zenodo.org/badge/537269386.svg)](https://zenodo.org/badge/latestdoi/537269386)

![Fonte_Logo](./fonte.png)

```
git clone git@github.com:coinse/fonte.git
```

|File|Description|
|------------------------------------|----------------|
|📄 [PREPRINT](./preprint.pdf) | Preprint of the paper |
|🖥 [REQUIREMENTS](./REQUIREMENTS.md)| HW/OS/SW requirements |
|🔨 [INSTALL](./INSTALL.md)| Installation guide |
|🪪 [LICENSE](LICENSE)|MIT license| 

**Fonte** is a technique for finding the commit that introduced a bug in a software project given a failure. The purpose of this research artifact is to provide the necessary information and instructions for using Fonte. This artifact would be useful for researchers or developers who are interested in analysing the commit history of software projects or identifying their bug inducing commits.

Included are:

- Instructions for setting up the environment, including required hardware and software (Python and Docker) and information on how to install dependencies
- Instructions for running Fonte, including an example command and a list of available arguments
- Instructions for reproducing the experiment results, including information on using a Jupyter notebook
- Optional instructions for extracting the core data for other Defects4J faults using a pre-built Docker image
- Information on the structure of the data directory and the location of the core data and BIC dataset

## **A. Environmental Setup**
- Hardware
  - Developed under Mac with Intel chip
  - Compatible with AMD64 processors
- Software
  - Tested with bash (recommended), zsh, PowerShell
  - Python 3.9+
    - If using `pyenv`, use these commands:
      ```bash
      pyenv install 3.9.1
      pyenv local 3.9.1
      ```
    - **Install dependencies**:
        ```bash
        pip install --upgrade pip
        python -m pip install numpy pandas scipy tqdm matplotlib seaborn rank-bm25 tabulate jupyter setuptools
        python -m pip install lib/SBFL
        # Alternative: python -m pip install git+https://github.com/Suresoft-GLaDOS/SBFL 
        python -m pip install lib/spiral
        # Alternative: python -m pip install git+https://github.com/casics/spiral
        ```
  - [Docker client](https://www.docker.com/products/docker-desktop) (only for the future extension)

## **B. Getting Started**

### Please be aware that
- Information on the file and directory structure can be found at the end of this README file.
- The necessary data including coverage matrix and commit history for each fault can be found in `./data/Defects4J/core/`
- The BIC dataset is located at `./data/Defects4J/BIC_dataset/`

### Running Fonte
```bash
python Fonte.py data/Defecst4J/core/<pid>-<vid>b -o <savepath>
```
- Example:
  ```bash
  python Fonte.py data/Defects4J/core/Cli-29b -o output.csv
  # Number of total commits: 616
  #          vote  rank  is_style_change
  # commit
  # c0d5c79   1.0   1.0            False
  # 147df44   0.0   2.0             True
  ```

- Available arguments:
  ```bash
  usage: Fonte.py [-h] [--tool TOOL] [--formula FORMULA] [--alpha ALPHA] [--tau TAU] [--lamb LAMB] [--skip-stage-2] [--output OUTPUT] coredir

  Compute commit scores

  positional arguments:
    coredir               data/Defects4J/core/<pid>-<vid>b

  optional arguments:
    -h, --help            show this help message and exit
    --tool TOOL           history retrieval tool, git or shovel (default: git)
    --formula FORMULA     SBFL formula (default: Ochiai)
    --alpha ALPHA         alpha: 0 or 1 (default: 0)
    --tau TAU             tau: max or dense (default: max)
    --lamb LAMB           lambda: [0.0, 1.0) (default: 0.1)
    --skip-stage-2        skip stage 2 (default: False)
    --output OUTPUT, -o OUTPUT
                          path to output file (example: output.csv)
  ```

## **C. Reproducing the experiment results**
1. Run the Jupyter notebook
    ```bash
    jupyter notebook
    ```
    If you're a VSCode user, just install the `Jupyter` extension.
2. Open `experiment.ipynb` and run the cells to reproduce our experiment results.
    - The output will be saved to `./experiment_results/`. Note that the directory already contains the pre-computed results. If you want to fully replicate our experiments, remove all files from the `./experiment_results/` and run the cells again.

## **D. Extension for other Defects4J faults** (Optional) 
To support further extension, we have created a pre-built Docker image that includes our data collection scripts and a fully installed version of Defects4J. This image can be used to extract the core data, in the same format as found in ./data/Defects4J/core/, for any Defects4J fault.

1. Pull the image from DockerHub. This may take a while because the image size is about 4GB.
    ```bash
    docker pull agb94/fonte:latest
    ```
2. Start a Docker container
    ```bash
    docker run -dt --name fonte -v $(pwd)/docker/workspace:/root/workspace agb94/fonte:latest
    ```
    - The directory `./docker/workspace` in the local machine will share data with `/root/workspace` in the container.
    - `$(pwd)`: The current directory. Change it to `${PWD}` or `%cd%` if you're using PowerShell or Windows Command Prompt, respectively.
3. Collect the coverage information and the commit history of `<pid>-<vid>b`
    ```bash
    docker exec fonte sh collect_core.sh <pid> <vid> <tool:git,shovel>
    # Example: docker exec fonte sh collect_core.sh Cli 29 git
    ```
    - The output will be saved to `./docker/workspace/data/<pid>-<vid>b/`
    - Don't forget to append the tool option (`git` or `shovel`)!
4. Run Fonte on the newly collected data:
    ```bash
    python Fonte.py ./docker/workspace/data/<pid>-<vid>b/
    # Example: python Fonte.py ./docker/workspace/data/Cli-29b/
    ```
    💡 To **speed up** the AST comparison, you can disable code formatting using OpenRewrite by appending `false` when calling the `collect_core.sh` script:
    
    ```bash
    docker exec fonte sh collect_core.sh <pid> <vid> <tool:git,shovel> false
    # docker exec fonte sh collect_core.sh Cli 29 git false
    ```

## **File & Directory Structure**
- `CodeShovel-error.md`: this contains the error reproduction steps for CodeShovel
- **`Fonte.py`: Fonte CLI**
- **`experiment.ipynb`: the main experiment script**
- `run_Bug2Commit.py`: the python script implementing Bug2Commit (not contained in the lightweight version)
- `data/`
  - `Defects4J/`
    - `BIC_dataset/`
      - `*.csv`
      - `README.md`: See this for more information about the BIC dataset
    - `core/`
      - `<pid>-<vid>b`
        - `git/`
          - `commits.pkl`: pandas Dataframe
          - `validation.csv`: style change commit validation results in the following format:
            ```csv
            commit,filepath,[C|U|N|E]
            ...
            ```
            - where
              - `C`: ASTs are different (**c**hanged)
              - `U`: ASTs are identical (**u**nchanged)
              - `N`: The file is **n**ewly introduced
              - `E`: External **e**rror by GumTree
            - A commit is a style-change commit only when the result for every related file is `U`.
          - `validation_noOpenRewrite.csv`: style change commit validation results without the CheckStyle fixes using OpenRewrite
        - `shovel/`
          - `raw/`: the raw output files from CodeShovel
          - `commits.pkl`: pandas Dataframe
          - `validation.csv`
          - `validation_noOpenRewrite.csv`: style change commit validation results without the CheckStyle fixes using OpenRewrite
        - `coverage.pkl`: pandas Dataframe (index: tests, columns: lines)
        - `commits.log`: all commits in the branch
        - `failing_tests`: the exception messages and stack traces of failing test cases (used when running Bug2Commit)
    - `baseline/`: this contains the ingredients & results for Bug2Commit and FBL-BERT.
       - `<pid>-<vid>b/`
         - `commits/`: the raw contents of commits that modified at least one `.java` file (collected using the `data_utils.py` in [FBL-BERT](https://anonymous.4open.science/r/fbl-bert-700C), not contained in the lightweight version)
         - `br_short.txt`: a title of the bug report
         - `br_long.txt`: a main body of the bug report
         - `ranking_INDEX_FBLBERT_RN_bertoverflow_QARC_q256_d230_dim128_cosine_q256_d230_dim128_commits_token.tsv`: raw retrieval results for `<pid>-<vid>b` (**FBL-BERT**)
         - `ranking_Bug2Commit.csv`: raw retrieval results for `<pid>-<vid>b` (**Bug2Commit**)
    - `buggy_methods.json`: The buggy method information is constructed for the actual buggy versions of programs in Defects4J (corresponding to `revision.id.buggy`). The actual buggy version may differ from the isolated buggy version provided by Defects4J that you obtain right after the checkout. Therefore, the buggy methods may not exactly match the methods fixed by the patch.
  - `industry/`: the results of applying Fonte to the batch testing data of an industry software ABC
    - `<data>_<test>.csv`: test names are anonymized due to DBR
- `docker/`: containing the docker resources that can be used to extract the core data
  - `resources/`: the resources needed to build the image from scratch
  - `Dockerfile`: the docker config file used to build the image `bic:new`
  - `workspace/`
    - `collect_core.sh`: the main script for code data extraction
    - `collect.py`
- `experiment_results/`
  - `rankings/`
    - `git_line_Ochiai_voting(_C_BIC|_C_susp)/`: the postfix `_C_susp` means skipping Stage 2
      - `<tau>-<alpha>-<lambda>.csv`
      - `score-<lambda>.csv`: baseline
      - `equal-<lambda>.csv`: baseline
    - `git_line_Ochiai_maxArrg(_C_BIC|_C_susp).csv`: max-aggregation baseline
    - `Random(_C_BIC|_C_susp).csv`: Random baseline
    - `FBL-BERT(_C_BIC|_C_susp).csv`: FBL-BERT results
    - `Bug2Commit(_C_BIC|_C_susp).csv`: Bug2Commit results
    - `Worst(_C_BIC|_C_susp).csv`: Lower bound of the results
- `lib/`
  - `SBFL/`
  - `spiral/`
  - `experiment_utils.py`: it contains the main functions
  - `README.md`
