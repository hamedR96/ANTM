def slice_by_year(list_zip, min_year, max_year, nb_years, overlap_years) :
    if(nb_years < overlap_years) :
        return []
    sliced_list_id = []
    sliced_list_dv = []
    list_tmp_id = []
    list_tmp_dv = []
    curr_year = min_year
    while (curr_year < max_year) :
        period = [curr_year + i for i in range(0,nb_years)]
        for i in range(len(list_zip)) :
            if list_zip[i][0] in period :
                list_tmp_id.append(list_zip[i][1])
                list_tmp_dv.append(list_zip[i][2])
        sliced_list_id.append(list_tmp_id)
        sliced_list_dv.append(list_tmp_dv)
        list_tmp_id = []
        list_tmp_dv = []
        curr_year = curr_year + nb_years - overlap_years
    return sliced_list_dv, sliced_list_id

def slice_df(df, t1, t2, w, o):
    slices = []
    slice_start = t1
    slice_end = t1 + w
    slice_num=0
    while slice_end <= t2+1:
        slice_num+=1
        slice = df[(df['time'] >= slice_start) & (df['time'] < slice_end)]
        slice_copy = slice.copy()
        slice_copy.loc[:, "slice_num"] = slice_num
        slice = slice_copy
        slices.append(slice)
        slice_start += w - o
        slice_end += w -o
    return slices

def relation_periodes(list_ids1, list_ids2) :
    D = dict()
    for i in list_ids1 :
        if i in list_ids2 :
            D[list_ids1.index(i)] = list_ids2.index(i)
    return D

def relations_periodes(list_list_id) :
    L = []
    for i in range(len(list_list_id)-1) :
        L.append(relation_periodes(list_list_id[i],list_list_id[i + 1]))
    return L

def sws(df_embedded,overlap,window_length):
    df_triple = [list(x) for x in zip(df_embedded.time.tolist(), df_embedded.index.tolist(), df_embedded.embedding.tolist())]
    slices=slice_df(df_embedded, df_triple[0][0],  df_triple[len(df_triple)-1][0], window_length,overlap)
    sliced_list_doc_vect, sliced_list_id = slice_by_year(df_triple,df_triple[0][0],
                                                               df_triple[len(df_triple)-1][0],window_length,overlap)
    arg1_umap = sliced_list_doc_vect
    arg2_umap = relations_periodes(sliced_list_id)
    return slices,arg1_umap,arg2_umap