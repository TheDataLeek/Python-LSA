%LSA using SVD on big matrix

clc;
clear all;

t = load('output_small.mat')
wordlist = t.words;   
u = t.u;
vt = t.vt;
d = t.d;
documents = t.documents;

k = 100;
query_flag = false;

num_docs = size(vt,2);
num_words = size(u,1);

word_mat = u * d;
doc_mat = d * vt;

for i = 1:num_words
    words{i} = word_mat(i,:);
end

for i = 1:num_docs
    docs{i} = doc_mat(:,i);
end

num_queries = input('Enter the number of words you wish to query');
for j = 1:num_queries
    fprintf('Query number %d:\n',j)
    query = input('Enter the query word');
    for i = 1:num_words
        query_comp = strcmpi(query,wordlist{i});
        if query_comp == 1
            query_flag = true;
            word_index(j) = i;
        end
    end 
    if query_flag == false
       errordlg('WTF enter a correct word');
       return;
    end
   query_flag = false;
end

q = zeros(1,k);
for i = 1:length(word_index)
    q = words{word_index(i)} + q;
end

q_avg = q/length(word_index);

for i = 1:num_docs
    rank(i) = dot(docs{i},q_avg)/norm(docs{i})*norm(q_avg);
end

[r I] = sort(rank,'descend');

