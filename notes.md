Our goal is to find the 5 neighborhoods with the largest spread in price. Our guy says he wants to go to this area, fix up old properties and flip them for a profit.

we cant just directly compare prices in a neighborhood, because first we need to group properties. like u cant compare a studio to a 4 bedroom house and say, wow look at the price difference!


So first step is some sort of clustering. 
Lets first split by 


Columns that for sure effect price, that arent revealing what the price is:
[10,11,12,16,18]



Columns that for sure dont effect price because they are irrelevant:




Columns that are too dirty to include:




Columns that are purely random (like an id):
1: MLS No


Columns that reveal price:
(for the first grouping we want it to be blind to price. we should group by all other data and then the second step is seeing which geographical area has the biggest price spread in the clustered data)



actually wait - before i do the price blind attribute clustering, we need to do a geographical clustering. 

state spread:
Number of unique values: 12
CA: 339715
AZ: 4
TX: 2
ID: 2
NV: 2
AL: 2
VA: 2
OR: 1
WA: 1
CO: 1
ME: 1
NY: 1

umm fuck it lets delete everything not CA first.

lets keep a folder with all the files and transformations we apply to them. 

Data Versions:
v0_combined_data.csv - 339746
    original dataset from ayush
v1_ca_only.csv - 339716 rows
    deleted everything not CA
v2_deduped_ca_only.csv - 330674 rows
    deduped by address, unit and city (columns 4 5 and 6), selected the most recent closing date for each duplicate group
v3_filtered_cities_ca_only.csv - 316909 rows
    filtered out cities with less than 100 rows


okay. now we kinda need the coordinates. so we want to do:

1. Geographical Clustering by lat/long
2. remove significant outliers  
3. Temporal Clustering by closing date
4. Attribute Clustering blind to price
5. Calculate price spread of each cluster.
6. Determine which geographical cluster has the highest price spread.

okay now lets also check for duplicates and dedupe

There are a total of 9041 duplicates checking by address, unit and city (columns 4 5 and 6)

okay now we deduped and everything is in CA. Now we want to create geographical clusters. first we need lat/long.

Consider time. time matters because valuation changes with time so we should also do temporal clustering.

So the struggle now is getting the lat/long. we have 330,674 rows. 

Google Maps Geocoding API:
4$ per 1000 requests
3000 QPM
Total cost: $1322.696
Total time: 1.83 hours

Radar Core Geocoding API:
100k free requests a month
After that 0.5$ per 1000 requests
10 QPS
Total cost: $115.337
Total time: 9.19 hours

Geocodio API
2500 free requests a day
After that 0.5$ per 1000 requests
1000 QPM
Total cost: $164.837
Total time: 5.51 hours

We also want to do this fast. google places api charges 4$ per 1000 geocoding requests. 

Okay so all options are either too expensive or too slow. 
What we can do instead is do the rest of our analysis without lat/long but do it at a city or zip level. then we can basically filter out a lot of the worst data and then just choose the rows that are in cities that look the most promising. 

Okay so how do we do temporal clustering? For each geographical cluster we can split it into 2 temporal clusters based on closing date - unless



okay now we'll do two runs through the playbook. we want to use long/lat but thats expensive asf, so we'll use cities as geographical clustering first, fo the rest of the steps, choose the best cities, then actually do long/lat on those and repeat the process with more accurate geographical clustering