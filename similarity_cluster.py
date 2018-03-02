import sys
from pyspark import SparkContext, SparkConf
from math import sqrt

def loadMovieNames():
    movieNames = {}
    with open('./ml-1m/movies.dat') as f:
        for line in f:
            fields = line.split('::')
            movieNames[int(fields[0])] = fields[1].decode('ascii', 'ignore')
    return movieNames

def readInputData():
    inputData = sc.textFile("wasb://sparkclusterarogau-2018-03-01t16-51-42-172z@utd9.blob.core.windows.net/root/sshuser/ml-1m/ratings.dat")
    ratings = inputData.map(lambda x: x.split('::')).map(lambda x: (int(x[0]),(int(x[1]),float(x[2]))))
    return ratings

def filterDuplicate(joinedRating):
    movie1 = joinedRating[1][0][0]
    movie2 = joinedRating[1][1][0]
    return movie1<movie2

def pairing(rating):
    (movie1, rating1) = rating[1][0]
    (movie2, rating2) = rating[1][1]    
    return ((movie1, movie2), (rating1,rating2))

def CosineSimilarity(ratingPairs):
    numPairs = 0
    xx = yy = xy = 0
    for ratingX, ratingY in ratingPairs:
        xx += ratingX * ratingX
        yy += ratingY * ratingY
        xy += ratingX * ratingY
        numPairs += 1

    numerator = xy
    denominator = sqrt(xx) * sqrt(yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)
    
# load movie names and their IDs
movieData = loadMovieNames() 

sc = SparkContext(conf = SparkConf())

#reading the , user, movie ID, and rating

ratings = readInputData()

#RDD => (user,(movieID, rating))

partitionedRatings = ratings.partitionBy(100)
joined = partitionedRatings.join(partitionedRatings)

#RDD => (user,((movieID, rating),(movieID, rating))) but with duplicates

filtered = joined.filter(filterDuplicate)

#RDD => (user,((movieID, rating),(movieID, rating))) duplicates removed

pairs = filtered.map(pairing).partitionBy(100)

#RDD => ((movie1, movie2),(rating1, rating2))

pairsGrouped = pairs.groupByKey()

#RDD => ((movie1, movie2),((rating1, rating2),(rating1, rating2)....)

similarity = pairsGrouped.mapValues(CosineSimilarity).persist()


# RDD => ((movie1,movie2),(score, numPairs))


if (len(sys.argv) > 1):
    simThreshold = 0.97
    coOccurence = 1000
    
    movieID = int(sys.argv[1])
    
    filteredResults = similarity.filter(lambda x: \
    (x[0][0] == movieID or x[0][1] == movieID) \
    & (x[1][0] > simThreshold) \
    & (x[1][1] > coOccurence))
    
    #sorting
    
    results = filteredResults.map(lambda x: (x[1],x[0])).sortByKey(ascending = False).take(10)
    
    for result in results:
        (sim,pair) = result
        similarMovieID = pair[0]
        if (similarMovieID == movieID):
            similarMovieID = pair[1]
        print(movieData[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))

