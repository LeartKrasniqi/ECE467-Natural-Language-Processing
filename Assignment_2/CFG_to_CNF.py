# Python script to convert CFG to CNF
#
# Assumes that all non-terminals start with a capital letter,
#   and all terminals start with a lowercase letter or digit
#
# In created CFG in CNF, newly created symbols will start with _
#
# I am not too worried about efficiency, just corretness

import sys

arrow = "-->"
orsymbol = "|"

def usage(argv):
    sys.stderr.write("\nUsage: python3 " + argv[0] + \
                     " <cfg_file> <cnf_file>\n\n");

def main(argv):
    numarg = len(argv)
    if numarg != 3:
        usage(argv)
        sys.exit(1)

    filenameIn = argv[1];
    filenameOut = argv[2];
    
    try:
        fileIn = open(filenameIn, "r")
    except:
        sys.stderr.write("\nError: Could not open input file: " + \
                         filenameIn + "\n")
        sys.exit(1)

    try:
        fileOut = open(filenameOut, "w")
    except:
        sys.stderr.write("\nError: Could not open output file: " + \
                         filenameOut + "\n")
        sys.exit(1)

    # Loop through lines of grammar
    sys.stdout.write("\nLoading grammar...\n")
    validRules = {}
    for line in fileIn:
        if line[0] == "#": # This line is a comment, skip it
            continue
        line = line[:-1]
        if not line.strip(): # ignore blank lines (only white space)
            continue
        tokens = line.split() # split by whitespace
        numTokens = len(tokens)

        # Perform some error checks, skip lines with errors
        isError = False
        if numTokens < 3:
            sys.stderr.write("Error: Too few tokens: " + line + "\n")
            isError = True
            continue

        if tokens[1] != arrow:
            sys.stderr.write("Error: Missing arrow: " + line + "\n")
            isError = True
            continue

        if not tokens[0][0].isupper():
            sys.stderr.write("Error: First token must be nonterminal, " + \
                             "start with captial letter: " + line + "\n")
            isError = True
            continue

        for i in range(numTokens):
            if i != 1 and tokens[i] == arrow:
                sys.stderr.write("Error: Extra arrow: " + line + "\n")
                isError = True
                break

            if (i == 0 and tokens[i] == orsymbol) or \
               (i == 2 and tokens[i] == orsymbol) or \
               (i == numTokens-1 and tokens[i] == orsymbol):
                sys.stderr.write("Error: Misplaced OR: " + line + "\n")
                isError = True
                break

            if i > 1 and \
               not tokens[i] == orsymbol and \
               not tokens[i][0].isupper() and \
               not tokens[i][0].islower() and \
               not tokens[i][0].isdigit():
                sys.stderr.write("Error: Invalid token: " + tokens[i] + \
                                 "\n  in line: " + line + "\n")
                isError = True
                break

            if i > 2 and tokens[i] == orsymbol and tokens[i-1] == orsymbol:
                sys.stderr.write("Error: Empty clause: " + line + "\n")
                isError = True
                break

        if isError:
            continue

        # If we get here, the line had no detected errors
        # Process the line and store the detected rule(s)
        nonterminal = tokens[0] # this is the non-terminal on the lhs
        curStart = 2
        # Split the rule in case of OR symbol(s)
        for i in range(2,numTokens+1):
            if i == numTokens or tokens[i] == orsymbol:
                # Valid rhs spans curStart to i-1
                rhs = tokens[curStart : i]

                # Store rule
                if not nonterminal in validRules.keys():
                    validRules[nonterminal] = []
                if not rhs in validRules[nonterminal]:
                    validRules[nonterminal].append(rhs)
                
                curStart = i+1                

    # Close input file
    fileIn.close()

    # All valid rules are now in the validRules dictionary
    #
    # For each non-terminal with at least one valid rule,
    #   that valid rule maps to a list of right-hand sides (a list of lists),
    #   where each rhs is a list of tokens (terminals and non-terminals)

    # Display valid rules in initial CFG (comment or uncomment as desired)
    # sys.stdout.write("\n")
    # for key in validRules.keys():
    #     keyRules = validRules[key]
    #     for rhs in keyRules:
    #         # Display valid rule
    #         sys.stdout.write("Valid CFG rule: " + key + " -->")
    #         for j in rhs:
    #             sys.stdout.write(" " + j)
    #         sys.stdout.write("\n")

    # Convert CFG (initial set of valid rules) to CNF
    sys.stdout.write("Converting to CNF...\n")
    cnfRules = {}
    numDummy = 0
    for key in validRules.keys():
        keyRules = validRules[key]
        for rhs in keyRules:
            # A valid rule in the original CFG is: key --> rhs

            # Check if already in CNF format
            if (len(rhs) == 1 and rhs[0][0].islower()) or \
               (len(rhs) == 1 and rhs[0][0].isdigit()) or \
               (len(rhs) == 2 and rhs[0][0].isupper() and rhs[1][0].isupper()):
                # Store rule (aleady valid in CNF)
                if not key in cnfRules.keys():
                    cnfRules[key] = []
                if not rhs in cnfRules[key]:
                    cnfRules[key].append(rhs)

            else:
                # Check if non-valid rule has terminal symbol
                containsTerminal = False
                for i in range(len(rhs)):
                    token = rhs[i]
                    if token[0].islower():
                        # Convert terminal in longer rule to dummy
                        numDummy = numDummy + 1
                        newNT = "_Dummy" + str(numDummy)
                        # Create new rule: newNT --> token
                        cnfRules[newNT] = []
                        cnfRules[newNT].append([token])
                        # Change terminal to dummy non-terminal
                        rhs[i] = newNT

                        containsTerminal = True

                if containsTerminal:
                    # We have converted a rule
                    if len(rhs) == 2:
                        # A new CNF rule has been created
                        if not key in cnfRules:
                            cnfRules[key] = []
                        if not rhs in cnfRules[key]:
                            cnfRules[key].append(rhs)

    # Only cases left to fix are singletons and long rules

    # Loop through all rules as many times as needed to determine singletons
    # If A --> B and B --> C, must also store A --> C
    # Need to avoid infinite loops if also C --> A
    # Also if D --> A, need to avoid infinite loops starting from D

    singletons = {} # Stores all singletons for each non-terminal

    # Using doLoop to determine if another pass is needed
    # Start it as true, so at least one pass occurs
    # Set it to false at the start of the pass
    # If a new singleton is found, it will be set back to true
    # If an entire pass finds no new singleton, that is the final pass
    doLoop = True
    while doLoop:
        doLoop = False
        for key in validRules.keys():
            keyRules = validRules[key]
            for rhs in keyRules:
                # A valid rule in the original CFG is: key --> rhs
                if len(rhs) == 1 and rhs[0][0].isupper():
                    # We have a singleton

                    # Ignore rules of the form A --> A
                    if key == rhs[0]:
                        continue
                    
                    # Add rhs[0] to key's list if it is not already there
                    if not key in singletons.keys():
                        singletons[key] = []
                    if rhs[0] not in singletons[key]:
                        singletons[key].append(rhs[0])
                        # Now add newly detected rules to replace singleton
                        if rhs[0] in validRules.keys():
                            keyRulesNew = validRules[rhs[0]]
                            for rhsNew in keyRulesNew:
                                # New rule is key --> rhsNew
                                # Ignore it if the form is A --> A
                                if len(rhsNew) == 1 and key == rhsNew[0]:
                                    continue
                                if not rhsNew in validRules[key]:
                                    validRules[key].append(rhsNew)

                                # If new rule is in CNF format, store it
                                if (len(rhsNew) == 1 and \
                                    rhsNew[0][0].islower()) or \
                                    (len(rhsNew) == 1 and \
                                     rhsNew[0][0].isdigit()) or \
                                     (len(rhsNew) == 2 and \
                                      rhsNew[0][0].isupper() and \
                                      rhsNew[1][0].isupper()):
                                    # Store rule (aleady valid in CNF)
                                    if not key in cnfRules.keys():
                                        cnfRules[key] = []
                                    if not rhsNew in cnfRules[key]:
                                        cnfRules[key].append(rhsNew)
                                
                        doLoop = True

    # Display all singletons (comment or uncomment as desired)
    # sys.stdout.write("\n")
    # for key in singletons.keys():
    #     sys.stdout.write("Singletons for " + key + ":")
    #     for s in singletons[key]:
    #         sys.stdout.write(" " + s)
    #     sys.stdout.write("\n")

    # Display updated CFG rules (comment or uncomment as desired)
    # sys.stdout.write("\n")
    # for key in validRules.keys():
    #     keyRules = validRules[key]
    #     for rhs in keyRules:
    #         # Display valid rule
    #         sys.stdout.write("Valid TMP rule: " + key + " -->")
    #         for j in rhs:
    #             sys.stdout.write(" " + j)
    #         sys.stdout.write("\n")
       
    # Only cases left to fix are long rules

    for key in validRules.keys():
        keyRules = validRules[key]
        for rhs in keyRules:
            # A valid rule in the original CFG is: key --> rhs
            if len(rhs) > 2:
                # This is a long rule that needs to be converted
                rhsTmp = rhs
                while len(rhsTmp) > 2:
                    # Create new dummy NT for first two NTs in rhsTmp
                    numDummy = numDummy + 1
                    newNT = "_Dummy" + str(numDummy)
                    # Create new rule: newNT --> rhsTmp[0] rhsTmp[1]
                    cnfRules[newNT] = []
                    cnfRules[newNT].append([rhsTmp[0], rhsTmp[1]])
                    # Replace first towo NTs in rhsTmp with dummy
                    rhsTmp = [newNT] + rhsTmp[2:]
                # Store final resulting rule
                if not key in cnfRules.keys():
                    cnfRules[key] = []
                if not rhsTmp in cnfRules[key]:
                    cnfRules[key].append(rhsTmp)

    # Display CNF rules (comment or uncomment as desired)
    # for key in cnfRules.keys():
    #     keyRules = cnfRules[key]
    #     for rhs in keyRules:
    #         # Display valid rule
    #         sys.stdout.write("Valid CNF rule: " + key + " -->")
    #         for j in rhs:
    #             sys.stdout.write(" " + j)
    #         sys.stdout.write("\n")

    # Output CNF rules
    for key in cnfRules.keys():
        keyRules = cnfRules[key]
        for rhs in keyRules:
            # Display valid rule
            fileOut.write(key + " -->")
            for j in rhs:
                fileOut.write(" " + j)
            fileOut.write("\n")

    sys.stdout.write("Done!\n")

if __name__ == "__main__":
    main(sys.argv)
