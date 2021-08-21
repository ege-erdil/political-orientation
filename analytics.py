import psycopg2
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
from scipy.stats import f_oneway

def connect():
    conn = psycopg2.connect(
       database="postgres", user='postgres', password='', host='', port= '5432'
    )
    return conn

def score_matrix(conn, question_list = []):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM result")
    results = cursor.fetchall()

    if question_list == []:
        cursor.execute("SELECT id FROM question WHERE \"isDeleted\"=FALSE")
        question_list = cursor.fetchall()
        question_list = [q[0] for q in question_list]

    cursor.execute("SELECT id, value FROM answer")
    answer_scores = cursor.fetchall()
    answer_scores = dict(answer_scores)

    user_list = list(set([x[3] for x in results]))

    scores = []
    for user in user_list:
        results_user = [x for x in results if x[3] == user]
        question_answer = sorted([(x[1], answer_scores[x[2]]) for x in results_user if x[1] in question_list], key=lambda y: y[0])
        if len(question_answer) == len(question_list):
            scores.append([x[1] for x in question_answer])

    return scores

def answer_matrix(conn, question_list = []):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM result")
    results = cursor.fetchall()

    if question_list == []:
        cursor.execute("SELECT id FROM question WHERE \"isDeleted\"=FALSE")
        question_list = cursor.fetchall()
        question_list = [q[0] for q in question_list]

    user_list = list(set([x[3] for x in results]))

    answers = []
    for user in user_list:
        results_user = [x for x in results if x[3] == user]
        question_answer = sorted([(x[1], x[2]) for x in results_user if x[1] in question_list], key=lambda y: y[0])
        if len(question_answer) == len(question_list):
            answers.append([x[1] for x in question_answer])

    return answers

def get_answer(conn, id_a):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM answer WHERE id = %d" % id_a)
    return cursor.fetchall()
    
def cov_matrix(scores):
    return np.cov(np.transpose(scores))

def tot_scores(scores):
    return [np.sum(scores[k]) for k in range(np.shape(scores)[0])]

def tot_factor_scores(scores, v):
    return [np.dot(scores[k], v) for k in range(np.shape(scores)[0])]

def question_correl(scores):
    (n, m) = np.shape(scores)
    scores_sum = tot_scores(scores)
    return sorted([(np.corrcoef(scores_sum, [x[j] for x in scores])[0, 1], j+1) if min([x[j] for x in scores]) < max([x[j] for x in scores]) else (0, j+1) for j in range(m)], key=lambda x: x[0])

def user_list(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM \"user\"")
    tab = cursor.fetchall()

    return [{"id": user[0], "age": user[1], "sex": user[2], "created": user[3], "email": user[4], "porientation": user[5]} for user in tab]
    

def safe_sqrt(x):
    if x >= 0:
        return np.sqrt(x)
    elif x > -1e-6:
        return 0
    else:
        return None

def anova(answers, scores):
    groups = []
    unique_answers = sorted(list(set(answers)))

    for j in range(len(unique_answers)):
        groups.append([])
        for k in range(len(answers)):
            if answers[k] == unique_answers[j]:
                groups[j].append(scores[k])

    return groups

def print_ans_list_q(conn, q_id):
    uniq_ans = sorted(list(set([x[q_id] for x in a])))
    for x in uniq_ans:
        print(get_answer(conn, x))
    
#establishing the connection
conn = connect()

#Creating a cursor object using the cursor() method
#cursor = conn.cursor()

#Executing an MYSQL function using the execute() method
#ursor.execute("SELECT * FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema'")
#cursor.execute("SELECT * FROM result")

#cursor.execute("SELECT * FROM information_schema.columns WHERE table_name = 'question'")

#m = score_matrix(conn, question_list=[x for x in range(1, 27) if x not in [1, 11, 14, 17, 21]])
m = score_matrix(conn)
a = answer_matrix(conn)
users = user_list(conn)

#u_a = [[u for u in users if u["porientation"] == j] for j in [2, 3, 4]]
#m_a = [[m[k] for k in range(len(m)) if users[k]["porientation"] == j] for j in [2, 3, 4]]
#a_a = [[a[k] for k in range(len(a)) if users[k]["porientation"] == j] for j in [2, 3, 4]]

#users = u_a[0] + u_a[1] + u_a[1] + u_a[2] + u_a[2]
#m = m_a[0] + m_a[1] + m_a[1] + m_a[2] + m_a[2]
#a = a_a[0] + a_a[1] + a_a[1] + a_a[2] + a_a[2]

#valid_users = [k for k in range(len(users)) if users[k]["age"] <= 60]

#m = [m[k] for k in valid_users]
#a = [a[k] for k in valid_users]
#users = [users[k] for k in valid_users]

tot = tot_scores(m)
tot_p = [tot[k] for k in range(len(users)) if users[k]["porientation"] in [2, 3, 4]]
ans_scores = [anova([x[j] for x in a], tot) for j in range(len(m[0]))]

c = cov_matrix(m)
w, v = np.linalg.eig(c)
print(len(m))

tot_factor = tot_factor_scores(m, v[:, 0])
tot_factor_p = [tot_factor[k] for k in range(len(users)) if users[k]["porientation"] in [2, 3, 4]]

wrh = sorted([(np.real(w[k]), k) for k in range(len(w))], key=lambda x: x[0])
wr = [x[0] for x in wrh]
wrn = wr/np.sum(wr)

print("Median score: %.3f" % (np.median(tot)))
print("Mean score: %.3f" % (np.mean(tot)))
print("Standard dev: %.3f" % (np.std(tot)))

print("Eigenvalues:")
for x in wrn:
    print("%.3f" % (100*x), end=", ")

print("\n")

v_arr = [sorted([(safe_sqrt(w[j]) * v[k, j]/(safe_sqrt(c[k, k])), k) if c[k, k] > 0 else (0, k) for k in range(np.shape(v)[0])], key=lambda x: x[0]) for j in range(np.shape(v)[1])]

for j in range(2):
    print("Factor %d:" % (j+1))
    for y in v_arr[j]:
        print("(%.3f, %d)" % (np.real(y[0]), y[1]+1), end=", ")
    print("")


print("Total score correlation:")
print(question_correl(m))

print("ANOVA tests:")
p_list = sorted([(f_oneway(*anova([x[j] for x in a], tot)).statistic * (len(anova([x[j] for x in a], tot)) - 1)/(len(tot) - 1), j+1) for j in range(np.shape(m)[1])], key=lambda x: x[0])
for p in p_list:
    print("(%.4f, %d)" % (p[0], p[1]), end=", ")

sr = [-1, 1, 0]
or_list = [sr[int(users[k]["porientation"])-2] for k in range(len(users)) if users[k]["porientation"] in [2, 3, 4]]

print("\nQuestion orientation correlations:")
or_correl = [(np.corrcoef(or_list, [m[j][k] for j in range(np.shape(m)[0]) if users[j]["porientation"] in [2, 3, 4]])[0, 1], k+1)
             for k in range(np.shape(m)[1]) if c[k, k] > 0]
or_correl = sorted(or_correl, key=lambda x: x[0])
print(or_correl)
print("")

genders = anova([users[k]["sex"] for k in range(len(users))], tot)
print("\nSex means")
print("Female: \t %.3f, error %.3f, std %.3f, num %d" %
      (np.mean(genders[0]), np.std(genders[0])/np.sqrt(len(genders[0])), np.std(genders[0]), len(genders[0])))
print("Male: \t \t %.3f, error %.3f, std %.3f, num %d" %
      (np.mean(genders[1]), np.std(genders[1])/np.sqrt(len(genders[1])), np.std(genders[1]), len(genders[1])))

print("\n")

porientations = anova([users[k]["porientation"] for k in range(len(users)) if users[k]["porientation"] in [1, 2, 3, 4]], [tot[k] for k in range(len(users)) if users[k]["porientation"] in [1, 2, 3, 4]])
#porientations = [[-7]] + porientations
print("Political orientation means")
print("Don't want to say: \t %.3f, error %.3f, std %.3f, num %d" %
      (np.mean(porientations[0]), np.std(porientations[0])/np.sqrt(len(porientations[0])), np.std(porientations[0]), len(porientations[0])))
print("Left-leaning: \t \t %.3f, error %.3f, std %.3f, num %d" %
      (np.mean(porientations[1]), np.std(porientations[1])/np.sqrt(len(porientations[1])), np.std(porientations[1]), len(porientations[1])))
print("Right-leaning: \t \t %.3f, error %.3f, std %.3f, num %d" %
      (np.mean(porientations[2]), np.std(porientations[2])/np.sqrt(len(porientations[2])), np.std(porientations[2]), len(porientations[2])))
print("Centrist: \t \t %.3f, error %.3f, std %.3f, num %d" %
      (np.mean(porientations[3]), np.std(porientations[3])/np.sqrt(len(porientations[3])), np.std(porientations[3]), len(porientations[3])))

print("\nPolitical orientation score correlation: %.3f" %
      np.corrcoef(or_list, tot_p)[0, 1])

print("\nPolitical orientation factor correlation: %.3f" %
      np.corrcoef(or_list, tot_factor_p)[0, 1])

tot_factor_arr = np.transpose(np.array([tot_factor_scores([m[k] for k in range(len(m)) if users[k]["porientation"] in [2, 3, 4]], v[:, k]) for k in range(2)]))
model = sm.OLS(endog=or_list, exog=sm.add_constant(tot_factor_arr))
res = model.fit()
print(res.summary())

#tot_factor_arr = np.transpose(np.array([tot_factor_scores([m[k] for k in range(len(m)) if users[k]["porientation"] in [2, 3, 4]], v[:, k]) for k in range(2)]))
model = MNLogit(endog=or_list, exog=sm.add_constant(tot_factor_arr))
res = model.fit()
print(res.summary())

psum = np.sum([len(porientations[k]) for k in [1, 2, 3]])
pshares = [len(porientations[k])/psum for k in [1, 2, 3]]
trivial_entropy = np.sum([-p * np.log(p) for p in pshares])
print("Model entropy %.3f, trivial entropy %.3f, information gain %.3f nats" % (-res.llf/res.nobs, trivial_entropy, res.llf/res.nobs + trivial_entropy))

def make_plot(res):
    pts = [0.1*k for k in range(-400, 400)]
    odds = [res.predict(exog=[1, k]) for k in pts]
    plt.plot(pts, [x[0][0] for x in odds], label="Left-leaning")
    plt.plot(pts, [x[0][1] for x in odds], label="Centrist")
    plt.plot(pts, [x[0][2] for x in odds], label="Right-leaning")
    plt.xlabel("Score")
    plt.ylabel("Probability")
    plt.legend()

    plt.show()

#make_plot(res)

#Closing the connection
conn.close()
