from anomaly import MStream

mstream = MStream([True, False, False])

mstream.learn_one({'c1': 3, 'n1': 1, 'n2': 8, 'timestamp': 1})
print(mstream.score_one({'c1': 3, 'n1': 1, 'n2': 8, 'timestamp': 1}))

mstream.learn_one({'c1': 4, 'n1': 2, 'n2': 8, 'timestamp': 2})
print(mstream.score_one({'c1': 4, 'n1': 2, 'n2': 8, 'timestamp': 2}))

mstream.learn_one({'c1': 4, 'n1': 1, 'n2': 8, 'timestamp': 2})
print(mstream.score_one({'c1': 4, 'n1': 1, 'n2': 8, 'timestamp': 2}))

o4 = {'c1': 5, 'n1': 3, 'n2': 7, 'timestamp': 3}
o5 = {'c1': 3, 'n1': 4, 'n2': 9, 'timestamp': 4}
o6 = {'c1': 4, 'n1': 1, 'n2': 9, 'timestamp': 5}
o7 = {'c1': 5, 'n1': 2, 'n2': 8, 'timestamp': 6}
o8 = {'c1': 20, 'n1': 20, 'n2': 20, 'timestamp': 7}

mstream.learn_one(o4)
print(mstream.score_one(o4))

mstream.learn_one(o5)
print(mstream.score_one(o5))

mstream.learn_one(o6)
print(mstream.score_one(o6))

mstream.learn_one(o7)
print(mstream.score_one(o7))

mstream.learn_one(o8)
print(mstream.score_one(o8))
