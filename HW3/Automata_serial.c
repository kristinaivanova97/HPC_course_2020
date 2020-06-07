//
//  gcc Automata.c
// ./a.out
// follow the instructions in command line

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

int rules (int a, int b, int c, int * ruleset){
    if      (a == 1 && b == 1 && c == 1) return ruleset[0];
    else if (a == 1 && b == 1 && c == 0) return ruleset[1];
    else if (a == 1 && b == 0 && c == 1) return ruleset[2];
    else if (a == 1 && b == 0 && c == 0) return ruleset[3];
    else if (a == 0 && b == 1 && c == 1) return ruleset[4];
    else if (a == 0 && b == 1 && c == 0) return ruleset[5];
    else if (a == 0 && b == 0 && c == 1) return ruleset[6];
    else if (a == 0 && b == 0 && c == 0) return ruleset[7];
    return 0;
}

//generate array with 0 and 1
void InitArray(int size, int * array)
{
    //random array
    //srand(time(NULL));
    //for(int i = 0; i < size; i++) array[i] = rand() % 2;
    
    // constant
    for(int i = 0; i < size; i++) array[i] = 0;
    array[size/2] = 1;
}

void generate(int size, int * prev_array, int * ruleset, int * new_array) {

    //constant boundary condition
    //new_array[0] = rules(1, prev_array[0], prev_array[1], ruleset);
    //new_array[size - 1] = rules(prev_array[size - 2], prev_array[size-1], 1, ruleset);
    
    // periodic boundary condition
    new_array[0] = rules(prev_array[size-1], prev_array[0], prev_array[1], ruleset);
    new_array[size - 1] = rules(prev_array[size - 2], prev_array[size-1], prev_array[0], ruleset);
    
    for (int i = 1; i < size - 1; i++) {
      int left   = prev_array[i-1];
      int me     = prev_array[i];
      int right  = prev_array[i+1];
      new_array[i] = rules(left, me, right, ruleset);
    }

  }

static void set_rule_binary(int rule, int * rule_binary)
{
    for(int p = 0; p <= 7; p++)
    {
        if((int)(pow(2, p)) & rule)
        {
            rule_binary[abs(p - 7)] = 1;
        }
        else
        {
            rule_binary[abs(p - 7)] = 0;
        }
    }
}


int main(int argc, char ** argv)
{
    
    int size;
    printf("Enter size of an array = ");
    scanf("%d", &size);
        
    //int * ruleset = (int*)malloc(8*sizeof(int));
    int rule;
    int * rule_binary = (int*)malloc(sizeof(int)*8);
    printf("Enter rule in Wolfram format = ");
    scanf("%d", &rule);
    set_rule_binary(rule, rule_binary);
    
    //print binary rule
    for (int i = 0; i < 8; i++) printf("%d ", rule_binary[i]);
    printf("\n");
    
    printf("order:  '1 1 1', '1 1 0', '1 0 1', '1 0 0', '0 1 1', '0 1 0', '0 0 1', '0 0 0'\n");
    /*
    printf("Enter eight zeros and ones as rules for stencils = \n");
    for(int i = 0; i < 8; i++)
    {
        scanf("%1d", &ruleset[i]);
    }
     */
     
    int rounds;
    printf("Enter number of repetitions = ");
    scanf("%d", &rounds);
        
    //printf("%d\n", rules(1, 0, 0, ruleset));
    
    int * array;

    array = (int*)malloc(size*sizeof(int));
    InitArray(size, array);
    printf("Initial array: \n");
    for (int i = 0; i < size; i++)
    {
       printf("%d", array[i]);
    }
    printf("\n");
    
    int * new_array = (int*)malloc(size*sizeof(int));

    // *** GENERATION PART ***
    for(int m = 0; m < rounds; m++)
    {
        generate(size, array, rule_binary, new_array); //or ruleset in place of rule_binary, if entered rules not in Wolfram style;
        //memcpy(&array, &new_array, size);

        for (int h = 0; h < size; h++)
        {
            array[h] = new_array[h]; //copy new_array in array to repeat process
        }

        for(int j = 0; j< size; j++)
        {
            if (new_array[j] == 0) {printf(" ");}
            else if (new_array[j] == 1) {printf("*");}
            else {printf("error ");}
        }
        printf("|");
        printf("\n");
        
    }

    free(array);
    //free(ruleset);
    free(new_array);
    free(rule_binary);
}

// IN serial version i'v got

/* for periodic boundary conditions
 rule: 0 1 0 1 1 0 1 0
 initial array:
 000000000000000100000000000000
              * *             |
             *   *            |
            * * * *           |
           *       *          |
          * *     * *         |
         *   *   *   *        |
        * * * * * * * *       |
       *               *      |
      * *             * *     |
     *   *           *   *    |
    * * * *         * * * *   |
   *       *       *       *  |
  * *     * *     * *     * * |
 *   *   *   *   *   *   *   *|
  * * * * * * * * * * * * * * |
 
same for static cond = 1:
 Initial array:
 000000000000000100000000000000
 *             * *            *|
 **           *   *          **|
  **         * * * *        ** |
  ***       *       *      *** |
  * **     * *     * *    ** * |
    ***   *   *   *   *  ***   |
 * ** ** * * * * * * * *** ** *|
 * ** **               * * ** *|
 * ** ***             *    ** *|
 * ** * **           * *  *** *|
 * **   ***         *   *** * *|
 * *** ** **       * * ** *   *|
 * * * ** ***     *    **  * **|
 *     ** * **   * *  *****  * |
 **   ***   *** *   ***   ***  |
  ** ** ** ** *  * ** ** ** ***|
  ** ** ** **  **  ** ** ** *  |
  ** ** ** ********** ** **  **|
 
 Initial array:
 Periodic boundaries.
 Rule : 1 0 1 1 0 1 1 0
 00000000000000000000000001000000000000000000000000
                         ***                       |
                        * * *                      |
                       *******                     |
                      * ***** *                    |
                     *** *** ***                   |
                    * * * * * * *                  |
                   ***************                 |
                  * ************* *                |
                 *** *********** ***               |
                * * * ********* * * *              |
               ******* ******* *******             |
              * ***** * ***** * ***** *            |
             *** *** *** *** *** *** ***           |
            * * * * * * * * * * * * * * *          |
           *******************************         |
          * ***************************** *        |
 
*/

/*
 Initial array randomly initialized:
 
1110011011100111100101001010000000111100
 * **  * * ** ** ***********     * ** **|
***  ******  *  * ********* *   ***  *  |
 * ** **** ******* ******* *** * * *****|
***  * ** * ***** * ***** * * ***** *** |
 * ****  *** *** *** *** ***** *** * * *|
*** ** ** * * * * * * * * *** * * ******|
** *  *  ***************** * ***** *****|
* ******* *************** *** *** * ****|
 * ***** * ************* * * * * *** ***|
*** *** *** *********** ********* * * * |
 * * * * * * ********* * ******* *******|
************* ******* *** ***** * ***** |
 *********** * ***** * * * *** *** *** *|
* ********* *** *** ******* * * * * * **|
 * ******* * * * * * ***** *********** *|
*** ***** *********** *** * ********* **|
** * *** * ********* * * *** ******* * *|
* *** * *** ******* ***** * * ***** *** |
** * *** * * ***** * *** ***** *** * * *|
* *** * ***** *** *** * * *** * * ***** |
 
*/
