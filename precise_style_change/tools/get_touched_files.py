import pandas as pd
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('commit_path', type=str)
  parser.add_argument('--commit', '-c', type=str)
  parser.add_argument('--output', '-o', type=str)

  args = parser.parse_args()
  df = pd.read_pickle(args.commit_path)
  if args.commit:
    touched = df['commit_hash'].apply(
      lambda sha: sha.startswith(args.commit)
    )
    df = df[touched]

  filtered_df = df[~df[['before_src_path', 'after_src_path']].apply(lambda row: '/dev/null' in row.values, axis=1)]
  files = list(filtered_df[['before_src_path', 'after_src_path']].drop_duplicates().itertuples(index=False, name=None))
  
  if args.output:
    with open(args.output, 'w') as f:
      for file_tuple in files:
        f.write(f'{file_tuple[0]} {file_tuple[1]}\n')
  else:
    for f in files:
      print(f)