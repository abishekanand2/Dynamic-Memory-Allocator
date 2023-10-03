/**
 * @file mm.c
 * @brief A 64-bit struct-based segregated free list memory allocator
 *
 * 15-213: Introduction to Computer Systems
 *
 * TODO: I have implemented a segregated free list allocator with the added optimizations of removing footers in allocated blocks
 * and mini-blocks (blocks of size 16 bytes). Below I describe my data structures.
 * 
 * Mini Blocks:
 * - Allocated: contain a header and an 8 byte payload
 * - Free: contain a header and a next pointer
 * - Mini free blocks are maintained in index 0 of my segregated list as a singly-linked list
 * 
 * Allocated Blocks (Non-mini):
 * - Contain a header and a payload
 * 
 * Free Blocks (Non-mini):
 * - Contain header, next/prev pointers, and footer
 * - These blocks are mainted in indices 1-14 of my segregated free list based on their size in doubly-linked lists.
 * 
 * Segregated Free Lists:
 * - 15-indexed array where each element is an explicit free list of a certain size class. Index 0 is for mini-blocks
 * and the rest of the indicies are blocks of size 2^i or less (where i is the index).
 *
 *************************************************************************
 *
 * ADVICE FOR STUDENTS.
 * - Step 0: Please read the writeup!
 * - Step 1: Write your heap checker.
 * - Step 2: Write contracts / debugging assert statements.
 * - Good luck, and have fun!
 *
 *************************************************************************
 *
 * @author Abishek Anand <abisheka@andrew.cmu.edu>
 */

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"

/* Do not change the following! */

#ifdef DRIVER
/* create aliases for driver tests */
#define malloc mm_malloc
#define free mm_free
#define realloc mm_realloc
#define calloc mm_calloc
#define memset mem_memset
#define memcpy mem_memcpy
#endif /* def DRIVER */

/* You can change anything from here onward */

/*
 *****************************************************************************
 * If DEBUG is defined (such as when running mdriver-dbg), these macros      *
 * are enabled. You can use them to print debugging output and to check      *
 * contracts only in debug mode.                                             *
 *                                                                           *
 * Only debugging macros with names beginning "dbg_" are allowed.            *
 * You may not define any other macros having arguments.                     *
 *****************************************************************************
 */
#ifdef DEBUG
/* When DEBUG is defined, these form aliases to useful functions */
#define dbg_requires(expr) assert(expr)
#define dbg_assert(expr) assert(expr)
#define dbg_ensures(expr) assert(expr)
#define dbg_printf(...) ((void)printf(__VA_ARGS__))
#define dbg_printheap(...) print_heap(__VA_ARGS__)
#else
/* When DEBUG is not defined, these should emit no code whatsoever,
 * not even from evaluation of argument expressions.  However,
 * argument expressions should still be syntax-checked and should
 * count as uses of any variables involved.  This used to use a
 * straightforward hack involving sizeof(), but that can sometimes
 * provoke warnings about misuse of sizeof().  I _hope_ that this
 * newer, less straightforward hack will be more robust.
 * Hat tip to Stack Overflow poster chqrlie (see
 * https://stackoverflow.com/questions/72647780).
 */
#define dbg_discard_expr_(...) ((void)((0) && printf(__VA_ARGS__)))
#define dbg_requires(expr) dbg_discard_expr_("%d", !(expr))
#define dbg_assert(expr) dbg_discard_expr_("%d", !(expr))
#define dbg_ensures(expr) dbg_discard_expr_("%d", !(expr))
#define dbg_printf(...) dbg_discard_expr_(__VA_ARGS__)
#define dbg_printheap(...) ((void)((0) && print_heap(__VA_ARGS__)))
#endif

/* Basic constants */

#define SEGLIST_LENGTH 15
typedef uint64_t word_t;

/** @brief Word and header size (bytes) */
static const size_t wsize = sizeof(word_t);

/** @brief Double word size (bytes) */
static const size_t dsize = 2 * wsize;

/** @brief Minimum block size (bytes) */
static const size_t min_block_size = dsize;

/**
 * Amount to extend heap by
 * (Must be divisible by dsize)
 */
static const size_t chunksize = (1 << 8);

/**
 * checks if block is alloc or free (when left-shifted it is used to check other flags)
 */
static const word_t alloc_mask = 0x1;

/**
 * used to extract the size 
 */
static const word_t size_mask = ~(word_t)0xF;

/** @brief Represents the header and payload of one block in the heap */
typedef struct block {
    word_t header;
    union {
        char payload[0];
        struct block *next_mini; // added for mini blocks
        struct {
            struct block *pred;
            struct block *succ;
        };
    };
} block_t;


/* Global variables */

/** @brief Pointer to first block in the heap */
static block_t *heap_start = NULL;

// array of explicit lists categorized by size
static block_t *seglist[SEGLIST_LENGTH];

/*
 *****************************************************************************
 * The functions below are short wrapper functions to perform                *
 * bit manipulation, pointer arithmetic, and other helper operations.        *
 *                                                                           *
 * We've given you the function header comments for the functions below      *
 * to help you understand how this baseline code works.                      *
 *                                                                           *
 * Note that these function header comments are short since the functions    *
 * they are describing are short as well; you will need to provide           *
 * adequate details for the functions that you write yourself!               *
 *****************************************************************************
 */

/*
 * ---------------------------------------------------------------------------
 *                        BEGIN SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/**
 * @brief Returns the maximum of two integers.
 * @param[in] x
 * @param[in] y
 * @return `x` if `x > y`, and `y` otherwise.
 */
static size_t max(size_t x, size_t y) {
    return (x > y) ? x : y;
}

/**
 * @brief Rounds `size` up to next multiple of n
 * @param[in] size
 * @param[in] n
 * @return The size after rounding up
 */
static size_t round_up(size_t size, size_t n) {
    return n * ((size + (n - 1)) / n);
}

/**
 * @brief Packs the `size` and `alloc` of a block into a word suitable for
 *        use as a packed value.
 * Edit: added prev_alloc and prev_is_mini to implement footerless and mini-blocks
 *
 * Packed values are used for both headers and footers.
 *
 * The allocation status is packed into the lowest bit of the word.
 *
 * @param[in] size The size of the block being represented
 * @param[in] alloc True if the block is allocated
 * @param[in] prev_alloc True if the prev block is allocated
 * @param[in] prev_is_mini True if the prev block is mini
 * @return The packed value
 */
static word_t pack(size_t size, bool alloc, bool prev_alloc, bool prev_is_mini) {
    word_t word = size;
    if (alloc) {
        word |= alloc_mask;
    }
    if (prev_alloc) {
        word |= (alloc_mask << 1);
    }
    if (prev_is_mini) {
        word |= (alloc_mask << 2);
    }
    return word;
}

/**
 * @brief Extracts the size represented in a packed word.
 *
 * This function simply clears the lowest 4 bits of the word, as the heap
 * is 16-byte aligned.
 *
 * @param[in] word
 * @return The size of the block represented by the word
 */
static size_t extract_size(word_t word) {
    return (word & size_mask);
}

/**
 * @brief Extracts the size of a block from its header.
 * @param[in] block
 * @return The size of the block
 */
static size_t get_size(block_t *block) {
    return extract_size(block->header);
}

/**
 * @brief Given a payload pointer, returns a pointer to the corresponding
 *        block.
 * @param[in] bp A pointer to a block's payload
 * @return The corresponding block
 */
static block_t *payload_to_header(void *bp) {
    return (block_t *)((char *)bp - offsetof(block_t, payload));
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        payload.
 * @param[in] block
 * @return A pointer to the block's payload
 * @pre The block must be a valid block, not a boundary tag.
 */
static void *header_to_payload(block_t *block) {
    dbg_requires(get_size(block) != 0);
    return (void *)(block->payload);
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        footer.
 * @param[in] block
 * @return A pointer to the block's footer
 * @pre The block must be a valid block, not a boundary tag.
 */
static word_t *header_to_footer(block_t *block) {
    dbg_requires(get_size(block) != 0 &&
                 "Called header_to_footer on the epilogue block");
    return (word_t *)(block->payload + get_size(block) - dsize);
}

/**
 * @brief Given a block footer, returns a pointer to the corresponding
 *        header.
 * @param[in] footer A pointer to the block's footer
 * @return A pointer to the start of the block
 * @pre The footer must be the footer of a valid block, not a boundary tag.
 */
static block_t *footer_to_header(word_t *footer) {
    size_t size = extract_size(*footer);
    dbg_assert(size != 0 && "Called footer_to_header on the prologue block");
    return (block_t *)((char *)footer + wsize - size);
}

/**
 * @brief Returns the payload size of a given block.
 *
 * The payload size is equal to the entire block size minus the sizes of the
 * block's header. (footerless)
 *
 * @param[in] block
 * @return The size of the block's payload
 */
static size_t get_payload_size(block_t *block) {
    size_t asize = get_size(block);
    return asize - wsize;
}

/**
 * @brief Returns the allocation status of a given header value.
 *
 * This is based on the lowest bit of the header value.
 *
 * @param[in] word
 * @return The allocation status correpsonding to the word
 */
static bool extract_alloc(word_t word) {
    return (bool)(word & alloc_mask);
}

/**
 * @brief Returns the allocation status of a block, based on its header.
 * @param[in] block
 * @return The allocation status of the block
 */
static bool get_alloc(block_t *block) {
    return extract_alloc(block->header);
}

/**
 * @brief Writes an epilogue header at the given address.
 *
 * The epilogue header has size 0, and is marked as allocated.
 *
 * @param[out] block The location to write the epilogue header
 */
static void write_epilogue(block_t *block, bool prev_alloc, bool is_prev_mini) {
    dbg_requires(block != NULL);
    dbg_requires((char *)block == (char *)mem_heap_hi() - 7);
    block->header = pack(0, true, prev_alloc, is_prev_mini);
}

/**
 * @brief Writes a block starting at the given address.
 *
 * This function writes both a header and footer, where the location of the
 * footer is computed in relation to the header.
 * 
 * Edit: changes were made to this function accordingly given the changes make to "pack"
 *
 *
 * @param[out] block The location to begin writing the block header
 * @param[in] size The size of the new block
 * @param[in] alloc The allocation status of the new block
 */
static void write_block(block_t *block, size_t size, bool alloc,
                        bool prev_alloc, bool is_prev_mini) {
    dbg_requires(block != NULL);
    // dbg_requires(size > 0);
    block->header = pack(size, alloc, prev_alloc, is_prev_mini);
    if (!alloc && size > 16) {
        word_t *footerp = header_to_footer(block);
        *footerp = pack(size, alloc, prev_alloc, is_prev_mini);
    }
}

/**
 * @brief Finds the next consecutive block on the heap.
 *
 * This function accesses the next block in the "implicit list" of the heap
 * by adding the size of the block.
 *
 * @param[in] block A block in the heap
 * @return The next consecutive block on the heap
 * @pre The block is not the epilogue
 */
static block_t *find_next(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(get_size(block) != 0 &&
                 "Called find_next on the last block in the heap");
    return (block_t *)((char *)block + get_size(block));
}

/**
 * @brief Finds the footer of the previous block on the heap.
 * @param[in] block A block in the heap
 * @return The location of the previous block's footer
 */
static word_t *find_prev_footer(block_t *block) {
    // Compute previous footer position as one word before the header
    return &(block->header) - 1;
}

/**
 * @brief Finds the previous consecutive block on the heap.
 *
 * This is the previous block in the "implicit list" of the heap.
 *
 * If the function is called on the first block in the heap, NULL will be
 * returned, since the first block in the heap has no previous block!
 *
 * The position of the previous block is found by reading the previous
 * block's footer to determine its size, then calculating the start of the
 * previous block based on its size.
 *
 * @param[in] block A block in the heap
 * @return The previous consecutive block in the heap.
 */
static block_t *find_prev(block_t *block) {
    dbg_requires(block != NULL);
    word_t *footerp = find_prev_footer(block);

    // Return NULL if called on first block in the heap
    if (extract_size(*footerp) == 0) {
        return NULL;
    }

    return footer_to_header(footerp);
}

/*
 * ---------------------------------------------------------------------------
 *                        END SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/******** The remaining content below are helper and debug routines ********/

// returns true if the previous block is a mini block
static bool get_prev_mini(block_t *block) {
    return (block->header) & (alloc_mask << 2);
}

// returns true if the previous block is allocated
static bool get_prev_alloc(block_t *block) {
    return (block->header) & (alloc_mask << 1);
}

// finds the appropriate index within the seglist based on the size of the block
static int find_index(size_t size) {
    int index = 14;
    int s = 16;
    for (int i = 0; i < SEGLIST_LENGTH; i++) {
        if (size <= ((size_t)s)) {
            index = i;
            return index;
        }
        s = s << 1;
    }
    return index;
}

// finds the previous mini block in index 0 of seglist
static block_t *find_prev_mini_seg(block_t *block)
{
    block_t *first = seglist[0];
    if (first == block)
    {
        return NULL;
    }
    while(first->next_mini != block)
    {
        first = first->next_mini;
    }
    return first;
}

// if prev block is mini, returns that block
// pre: previous block is a mini block
static block_t *find_prev_mini(block_t *block)
{
    return (block_t *)((char *)block - 16);
}

// returns true if the block is a mini block
static bool get_block_mini(size_t size)
{
    if (size == 16)
    {
        return true;
    }
    return false;
}

// debugging function that prints heap information
static void print_heap()
{
    block_t *tmp;
    for (tmp = heap_start; get_size(tmp) > 0; tmp = find_next(tmp))
    {
        printf("address: %p ", (void*)tmp);
        printf("size: %zu ", get_size(tmp));
        printf("cur_alloc: %d ", get_alloc(tmp));
        printf("prev_alloc %d ", get_prev_alloc(tmp));
        printf("prev_mini %d ", get_prev_mini(tmp));
        printf("\n");
    }
}

// inserts a node into seglist
static void insert_node(block_t *block) {
    size_t size = get_size(block);
    int index = find_index(size);
    if (index == 0) // mini blocks
    {
        // Case 1: list is empty
        if (!seglist[index])
        {
            seglist[index] = block;
            block->next_mini = NULL;
        }
        // Case 2: list is nonempty
        else
        {
            block->next_mini = seglist[index];
            seglist[index] = block;
        }
    }
    else
    {
        // Case 1: list is empty
        if (!seglist[index]) 
        {
            seglist[index] = block;
            block->succ = NULL;
            block->pred = NULL;
        } 
        // Case 2: list is nonmpty
        else 
        {
            block->succ = seglist[index];
            seglist[index]->pred = block;
            block->pred = NULL;
            seglist[index] = block;
        }
    }
    
}

// removes a node from seglist
static void remove_node(block_t *block) {
    assert(block != NULL);
    size_t size = get_size(block);
    int index = find_index(size);
    if (index == 0) // mini blocks
    {
        block_t *next_mini = block->next_mini;
        block_t *prev_mini = find_prev_mini_seg(block);
        if (prev_mini == NULL && next_mini != NULL) {
            seglist[index] = next_mini;
        } else if (next_mini == NULL && prev_mini != NULL) {
            prev_mini->next_mini = NULL;
        } else if (prev_mini == NULL && next_mini == NULL) {
            seglist[index] = NULL;
        } else {
            assert(prev_mini != NULL);
            assert(next_mini != NULL);
            prev_mini->next_mini = next_mini;
        }
        
    }
    else
    {
        block_t *next = block->succ;
        block_t *prev = block->pred;
        if (prev == NULL && next != NULL) {
            next->pred = NULL;
            seglist[index] = next;
        } else if (next == NULL && prev != NULL) {
            assert(prev != NULL);
            prev->succ = NULL;
        } else if (prev == NULL && next == NULL) {
            seglist[index] = NULL;
        } else {
            assert(prev != NULL);
            assert(next != NULL);
            prev->succ = next;
            next->pred = prev;
        }
    }
}

/**
 * If consecutive blocks are free, coalesce joins them together
 * @param[in] block
 * @return
 */
static block_t *coalesce_block(block_t *block) {
    dbg_requires(mm_checkheap);
    bool is_prev_mini = get_prev_mini(block);
    size_t is_prev_alloc = get_prev_alloc(block);
    size_t is_next_alloc = get_alloc(find_next(block));
    size_t size = get_size(block);

    // Case 1 (both prev and next are allocated)
    if (is_prev_alloc && is_next_alloc) {
        insert_node(block);
        write_block(block, size, false, true, is_prev_mini);
        write_block(find_next(block), get_size(find_next(block)), true, false, get_block_mini(size));
    }

    // Case 2 (prev allocated but next is free)
    else if (is_prev_alloc && !is_next_alloc) {
        remove_node(find_next(block));
        size += get_size(find_next(block));
        write_block(block, size, false, true, is_prev_mini);
        write_block(find_next(block), get_size(find_next(block)), true, false, get_block_mini(size));
        insert_node(block);
    }

    // Case 3 (prev free but next allocated)
    else if (!is_prev_alloc && is_next_alloc) {
        block_t *prev;
        if (is_prev_mini)
        {
            prev = find_prev_mini(block);
        }
        else
        {  
            prev = find_prev(block);
        }
        remove_node(prev);
        size += get_size(prev);
        write_block(prev, size, false, true, get_prev_mini(prev));
        write_block(find_next(block), get_size(find_next(block)), true, false, get_block_mini(size));
        block = prev;
        insert_node(block);
    }

    // Case 4 (both prev and next are free)
    else {
        block_t *prev;
        if (is_prev_mini)
        {
            prev = find_prev_mini(block);
        }
        else
        {  
            prev = find_prev(block);
        }
        remove_node(prev);
        remove_node(find_next(block));
        size += get_size(prev) + get_size(find_next(block));
        write_block(prev, size, false, true, get_prev_mini(prev));
        block = prev;
        write_block(find_next(block), get_size(find_next(block)), true, false, get_block_mini(size));
        insert_node(block);
    }

    return block;
}

/**
 * @brief
 *
 * <What does this function do?>
 * Extends the heap if there are no free blocks large enough for the allocation request
 * <What are the function's arguments?>
 * Takes in the size to extend by (chunksize)
 * <What is the function's return value?>
 * Returns the free block associated with the memory returned from mem_sbrk
 *
 * @param[in] size
 * @return
 */
static block_t *extend_heap(size_t size) {
    void *bp;

    // Allocate an even number of words to maintain alignment
    size = round_up(size, dsize);
    if ((bp = mem_sbrk((intptr_t)size)) == (void *)-1) {
        return NULL;
    }

    /*
     * TODO: delete or replace this comment once you've thought about it.
     * Think about what bp represents. Why do we write the new block
     * starting one word BEFORE bp, but with the same size that we
     * originally requested?
     */

    // Initialize free block header/footer
    block_t *block = payload_to_header(bp);
    bool prev_alloc1 = get_prev_alloc(block);
    bool prev_mini = get_prev_mini(block);
    write_block(block, size, false, prev_alloc1, prev_mini);

    // Create new epilogue header
    bool is_block_mini = get_block_mini(size);
    block_t *block_next = find_next(block);
    write_epilogue(block_next, false, is_block_mini);

    // Coalesce in case the previous block was free
    block = coalesce_block(block);

    return block;
}

/**
 * @brief
 *
 * <What does this function do?>
 * If there is free space left in a block after a portion of it is allocated, the block is split
 * <What are the function's arguments?>
 * The newly allocated block and the size of the block that is actually allocated
 * <What is the function's return value?>
 * Void
 * <Are there any preconditions or postconditions?>
 * Input block must be allocated upon entry and return
 *
 * @param[in] block
 * @param[in] asize
 */
static void split_block(block_t *block, size_t asize) {
    dbg_requires(get_alloc(block));

    size_t block_size = get_size(block);
    remove_node(block);
    if ((block_size - asize) >= min_block_size) {
        block_t *block_next;
        bool prev_alloc = get_prev_alloc(block);
        bool prev_mini = get_prev_mini(block);
        write_block(block, asize, true, prev_alloc, prev_mini);
        bool is_block_mini = get_block_mini(asize);

        block_next = find_next(block);
        write_block(block_next, block_size - asize, false, true, is_block_mini);
        insert_node(block_next);
    } else {
        bool is_block_mini = get_block_mini(block_size);
        write_block(find_next(block), get_size(find_next(block)),
                    get_alloc(find_next(block)), true, is_block_mini);
    }

    dbg_ensures(get_alloc(block));
}

/**
 * @brief
 *
 * <What does this function do?>
 * Finds a block that fits the allocation request using a "better" fit algorithm
 * - Implements "best fit" for the first 10 free blocks, then first fit after that
 * Returns NULL if no block is found
 * <What are the function's arguments?>
 * allocation request size
 * <What is the function's return value?>
 * returns a block that fits the request size
 *
 * @param[in] asize
 * @return
 */
static block_t *find_fit(size_t asize) {
    int index = find_index(asize);
    block_t *res = NULL;
    int count = 0;
    while (index < SEGLIST_LENGTH) {
        block_t *block = seglist[index];
        while (block != NULL) {
            if (get_size(block) >= asize) {
                if (count < 11)
                {
                    if (res == NULL || (get_size(block) < get_size(res)))
                    {
                        res = block;
                    }
                }
                else
                {
                    return block;
                }
            }
            count++;
            if (count > 10 && res != NULL)
                return res;
            if (index == 0)
            {
                block = block->next_mini;
            }
            else
            {  
                block = block->succ;
            }
        }
        index++;
    }
    return res;
}

// helper function for checkheap
// checking if blocks escaped coalescing
static bool checkCoalesce(block_t *tmp, int line) {
    if (heap_start == NULL) {
        return true;
    }
    block_t *first = heap_start;
    assert(first != NULL);
    block_t *next = find_next(first);

    while (next != NULL && get_size(next) > 0) {
        if (!get_alloc(first) && !get_alloc(next)) {
            return false;
        }
        first = find_next(first);
        next = find_next(next);
    }
    return true;
}

// helper function for checkheap
// check alignment and if header/footer match
static bool checkBlock(block_t *tmp, int line) {
    size_t address = (size_t)header_to_payload(tmp);
    if (address % 16 != 0) {
        dbg_printf("address: %zu \n", address % 16);
        dbg_printf("block address not aligned at line %d \n", line);
        return false;
    }

    if (!((void *)tmp >= mem_heap_lo()) || !((void *)tmp <= mem_heap_hi())) {
        dbg_printf("block address not within heap bounds at line %d \n", line);
        return false;
    }
    return true;
}

/**
 * @brief
 *
 * <What does this function do?>
 * Performs some basic checks on the state of the heap
 * <What are the function's arguments?>
 * line number
 * <What is the function's return value?>
 * true if no errors, false if error
 *
 * @param[in] line
 * @return
 */
bool mm_checkheap(int line) {
    block_t *tmp;
    int free_count1 = 0;
    int free_count2 = 0;
    for (tmp = heap_start; get_size(tmp) > 0; tmp = find_next(tmp)) {
        if (!checkCoalesce(tmp, line)) {
            return false;
        }

        if (!checkBlock(tmp, line)) {
            return false;
        }
        if (!get_alloc(tmp))
        {
            free_count1++;
        }
    }

    int i = 0;
    while (i < SEGLIST_LENGTH)
    {
        for (tmp = seglist[i]; tmp != NULL; tmp =  tmp->succ)
        {
            free_count2++;
        }
    }

    if (free_count1 != free_count2)
    {
        dbg_printf("number of free blocks in heap doesn't match number of free blocks in seglist \n");
    }

    return true;
}

/**
 * @brief
 *
 * <What does this function do?>
 * initializes the heap
 * <What are the function's arguments?>
 * void
 * <What is the function's return value?>
 * true if success
 *
 * @return
 */
bool mm_init(void) {
    // Create the initial empty heap
    for (int i = 0; i < SEGLIST_LENGTH; i++) {
        seglist[i] = NULL;
    }
    word_t *start = (word_t *)(mem_sbrk(2 * wsize));

    if (start == (void *)-1) {
        return false;
    }

    start[0] = pack(0, true, true, true); // Heap prologue (block footer)
    start[1] = pack(0, true, true, true); // Heap epilogue (block header)

    // Heap starts with first "block header", currently the epilogue
    heap_start = (block_t *)&(start[1]);

    // Extend the empty heap with a free block of chunksize bytes
    if (extend_heap(chunksize) == NULL) {
        return false;
    }

    return true;
}

/**
 * @brief
 *
 * <What does this function do?>
 * allocates a block with the requested size
 * <What are the function's arguments?>
 * the requested size
 * <What is the function's return value?>
 * void
 * <Are there any preconditions or postconditions?>
 *
 * @param[in] size
 * @return
 */
void *malloc(size_t size) {
    dbg_requires(mm_checkheap(__LINE__));
    size_t asize;      // Adjusted block size
    size_t extendsize; // Amount to extend heap if no fit is found
    block_t *block;
    void *bp = NULL;

    // Initialize heap if it isn't initialized
    if (heap_start == NULL) {
        if (!(mm_init())) {
            return NULL;
        }
    }

    // Ignore spurious request
    if (size == 0) {
        dbg_ensures(mm_checkheap(__LINE__));
        return bp;
    }

    // Adjust block size to include overhead and to meet alignment requirements
    asize = round_up(size + wsize, dsize);

    // Search the free list for a fit
    block = find_fit(asize);

    // If no fit is found, request more memory, and then and place the block
    if (block == NULL) {
        // Always request at least chunksize
        extendsize = max(asize, chunksize);
        block = extend_heap(extendsize);
        // extend_heap returns an error
        if (block == NULL) {
            return bp;
        }
    }

    // The block should be marked as free
    dbg_assert(!get_alloc(block));

    // Mark block as allocated
    size_t block_size = get_size(block);
    bool prev_alloc = get_prev_alloc(block);
    bool is_prev_mini = get_prev_mini(block);
    write_block(block, block_size, true, prev_alloc, is_prev_mini);
    // if size is 16 change prev_mini status of next block
    if (get_block_mini(block_size))
    {
        block_t *next = find_next(block);
        write_block(next, get_size(next), get_alloc(next), true, true);
    }
    else
    {  
        block_t *next = find_next(block);
        write_block(next, get_size(next), get_alloc(next), true, false);
    }

    // Try to split the block if too large
    split_block(block, asize);

    bp = header_to_payload(block);

    dbg_ensures(mm_checkheap(__LINE__));
    return bp;
}

/**
 * @brief
 *
 * <What does this function do?>
 * frees an allocated block
 * <What are the function's arguments?>
 * pointer to the blocks payload
 * <What is the function's return value?>
 * void
 *
 * @param[in] bp
 */
void free(void *bp) {
    dbg_requires(mm_checkheap(__LINE__));

    if (bp == NULL) {
        return;
    }

    block_t *block = payload_to_header(bp);
    size_t size = get_size(block);

    // The block should be marked as allocated
    dbg_assert(get_alloc(block));

    // Mark the block as free
    bool prev_alloc = get_prev_alloc(block);
    bool is_prev_mini = get_prev_mini(block);
    write_block(block, size, false, prev_alloc, is_prev_mini);

    // Try to coalesce the block with its neighbors
    coalesce_block(block);

    dbg_ensures(mm_checkheap(__LINE__));
}

/**
 * @brief
 *
 * <What does this function do?>
 * <What are the function's arguments?>
 * <What is the function's return value?>
 * <Are there any preconditions or postconditions?>
 *
 * @param[in] ptr
 * @param[in] size
 * @return
 */
void *realloc(void *ptr, size_t size) {
    block_t *block = payload_to_header(ptr);
    size_t copysize;
    void *newptr;

    // If size == 0, then free block and return NULL
    if (size == 0) {
        free(ptr);
        return NULL;
    }

    // If ptr is NULL, then equivalent to malloc
    if (ptr == NULL) {
        return malloc(size);
    }

    // Otherwise, proceed with reallocation
    newptr = malloc(size);

    // If malloc fails, the original block is left untouched
    if (newptr == NULL) {
        return NULL;
    }

    // Copy the old data
    copysize = get_payload_size(block); // gets size of old payload
    if (size < copysize) {
        copysize = size;
    }
    memcpy(newptr, ptr, copysize);

    // Free the old block
    free(ptr);

    return newptr;
}

/**
 * @brief
 *
 * <What does this function do?>
 * <What are the function's arguments?>
 * <What is the function's return value?>
 * <Are there any preconditions or postconditions?>
 *
 * @param[in] elements
 * @param[in] size
 * @return
 */
void *calloc(size_t elements, size_t size) {
    void *bp;
    size_t asize = elements * size;

    if (elements == 0) {
        return NULL;
    }
    if (asize / elements != size) {
        // Multiplication overflowed
        return NULL;
    }

    bp = malloc(asize);
    if (bp == NULL) {
        return NULL;
    }

    // Initialize all bits to 0
    memset(bp, 0, asize);

    return bp;
}

/*
 *****************************************************************************
 * Do not delete the following super-secret(tm) lines!                       *
 *                                                                           *
 * 53 6f 20 79 6f 75 27 72 65 20 74 72 79 69 6e 67 20 74 6f 20               *
 *                                                                           *
 * 66 69 67 75 72 65 20 6f 75 74 20 77 68 61 74 20 74 68 65 20               *
 * 68 65 78 61 64 65 63 69 6d 61 6c 20 64 69 67 69 74 73 20 64               *
 * 6f 2e 2e 2e 20 68 61 68 61 68 61 21 20 41 53 43 49 49 20 69               *
 *                                                                           *
 * 73 6e 27 74 20 74 68 65 20 72 69 67 68 74 20 65 6e 63 6f 64               *
 * 69 6e 67 21 20 4e 69 63 65 20 74 72 79 2c 20 74 68 6f 75 67               *
 * 68 21 20 2d 44 72 2e 20 45 76 69 6c 0a c5 7c fc 80 6e 57 0a               *
 *                                                                           *
 *****************************************************************************
 */
