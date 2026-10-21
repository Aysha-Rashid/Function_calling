from src.algo import initialize_everything
#store the files in an object of a class 
#store all the content of the file from the function defination in iterable format
#then start the function to use for one prompt at a time for decontraint decode
#we also have to read the vocab map and store it as an object so it is iterable as well (dict format)
#we create a state machine so that we can track where we are at the json format and what needs to be constraint
#the state machine contains what happens after the function name, after the parameter name, the parameter value, if there is any other parameter key or value 
#and after the parameter value, indication of ending the json format
#what we have to consider is that function name being used properly and the parameter type and value are properly used which includes the number, string, bool type

def main() -> None:
    initialize_everything()
    #create a function to initialize the files
    #read all the files from the cli or makefile


if __name__ == "__main__":
    main()
