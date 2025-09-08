lectures = ["9:00-10:30", "9:30-11:30","11:00-12:00",
            "14:00-18:00", "15:00-16:00", "15:30-17:30", 
            "16:00-18:00"]

def find_rooms(lectures):
    n = len(lectures)
    
    # convert lectures to a tuples list
    lectures = [ x.split('-') for x in lectures]
    # get start hours 
    starts = [ x[0].split(':') for x in lectures ]
    # convert to minutes and sort
    starts = sorted([ int(x[0])*60+int(x[1]) for x in starts ])
    # get end hours 
    ends = [ x[1].split(':') for x in lectures ]
    # convert to minutes and sort
    ends = sorted([ int(x[0])*60+int(x[1]) for x in ends ])
    
    rooms = 1
    max_rooms = rooms
    i, j = 1, 0 # indices to loop over starts and ends
    
    while i < n and j < n:
        if starts[i] < ends[j]:  
            rooms += 1 
            i += 1
            max_rooms = max(rooms, max_rooms)    # Update max rooms needed 
        else:  
            rooms -= 1
            j += 1
               
    return max_rooms

print(f" We need {find_rooms(lectures)} rooms")
