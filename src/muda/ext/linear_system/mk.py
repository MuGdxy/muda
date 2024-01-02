import os
def create(filename):
    h_filename = filename + '.h'
    detail_filename = 'details/' + filename + '.inl'
    if not os.path.exists(h_filename):
        f = open(h_filename, 'w')
        content = f'''#pragma once


    
#include "details/{filename}.inl"
'''
        f.write(content)
        f.close()
    else:
        print(f'{h_filename} already exists')

    if not os.path.exists(detail_filename):
        f = open(detail_filename, 'w')
        f.write('')
        f.close()
    else:
        print(f'{detail_filename} already exists')


create('linear_system_context')
create('vector_format_converter')
create('matrix_format_converter')

create('device_dense_matrix')
create('dense_matrix_view')
create('dense_matrix_viewer')

create('device_dense_vector')
create('dense_vector_view')
create('dense_vector_viewer')

create('device_triplet_matrix')
create('triplet_matrix_view')
create('triplet_matrix_viewer')

create('device_doublet_vector')
create('doublet_vector_view')
create('doublet_vector_viewer')

create('device_bcoo_matrix')
create('bcoo_matrix_view')
create('bcoo_matrix_viewer')

create('device_bcoo_vector')
create('bcoo_vector_view')

create('device_bsr_matrix')
create('bsr_matrix_view')

create('device_csr_matrix')
create('csr_matrix_view')


