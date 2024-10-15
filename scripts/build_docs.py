from project_dir import project_dir
import argparse as ap
import mkdocs
import subprocess as sp

if __name__ == '__main__':
    muda_project_dir = project_dir()
    parser = ap.ArgumentParser(description='Build the documents')
    muda_doc_default_dir = muda_project_dir.parent / 'muda-doc' / 'docs'
    parser.add_argument('-o', '--output', help='the output dir', default=f'{muda_doc_default_dir}')
    args = parser.parse_args()
    muda_doc_dir = args.output
    print(f'output_dir={muda_doc_default_dir}')
    config_file = muda_project_dir / 'mkdocs.yaml'
    print(f'config_file={config_file}')
    Value = sp.call(['mkdocs', 'build', '-f', config_file, '-d', muda_doc_dir])
    if Value == 0:
        print('Success')
    else:
        print('Failure')