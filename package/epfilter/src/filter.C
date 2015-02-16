
#include <stdio.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#ifdef _ALMOS_
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#else
#include <sys/time.h>
#include <sys/sysinfo.h>
#endif
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

////////////////////////////////////
// Image parameters

#define PIXEL_SIZE     2

#define PRINTF(...)      ({ if (tid == 0) { printf(__VA_ARGS__); } })

#define TA(c,l,p)  (A[c][((np) * (l)) + (p)])
#define TB(c,p,l)  (B[c][((nl) * (p)) + (l)])
#define TC(c,l,p)  (C[c][((np) * (l)) + (p)])
#define TD(c,l,p)  (D[c][((np) * (l)) + (p)])
#define TZ(c,l,p)  (Z[c][((np) * (l)) + (p)])

#define max(x,y) ((x) > (y) ? (x) : (y))
#define min(x,y) ((x) < (y) ? (x) : (y))

#define DEFAULT_NTHREADS 1

MAIN_ENV;
BARDEC(common_barrier);
CLOCK_DEC;

const uint32_t pixel_size = PIXEL_SIZE; // Size of a pixel
int32_t nlocal_threads;
int32_t nglobal_threads;
int32_t nclusters;
uint32_t npixels;    // Number of pixel per frame
uint32_t frame_size; // Size of 1 frame (in bytes)
int32_t nl;
int32_t np;

// Arrays of pointers on the shared, distributed buffers  
// containing the images (sized for the worst case : 256 clusters)
uint16_t ** A;
int32_t  ** B;
int32_t  ** C;
int32_t  ** D;
uint8_t  ** Z;

char * img_name;


static void usage(char * name) {
    printf("Usage: %s <options> <image_name>\n", name);
    printf("options:\n");
    printf("   -nN : N = Number of threads (default = %d; 0: try to guess the number of processors).\n", DEFAULT_NTHREADS);
    printf("   -lL : L = Number of lines in image (no default, mandatory).\n");
    printf("   -cC : C = Number of columns in image (no default, mandatory).\n");
    printf("   -h  : Display help.\n");
}


void process(void * i);

int main(int argc, char ** argv) {
    int32_t i;
    char ch;

    CLOCK_APP_START;

    nl = -1;
    np = -1;

    nglobal_threads = DEFAULT_NTHREADS;
    while ((ch = getopt(argc, argv, "n:l:c:h")) != -1) {
        switch (ch) {
            case 'n': nglobal_threads = atol(optarg);
#ifndef _ALMOS_
                      if (nglobal_threads == 0)
                          nglobal_threads = get_nprocs();
#endif
                      break;
            case 'l':
                      nl = atoi(optarg);
                      break;
            case 'c':
                      np = atoi(optarg);
                      break;
            case 'h': 
                      usage(argv[0]);
                      exit(0);
                      break;
            default:
                      usage(argv[0]);
                      exit(1);
                      break;
        }
    }

    if (np == -1 || nl == -1) {
        usage(argv[0]);
        exit(1);
    }

    i = 1;
    while (i < argc && argv[i][0] == '-') {
        i++;
    }

    if (i == argc) {
        usage(argv[0]);
        exit(1);
    }

    CLOCK_INIT(nglobal_threads);

    img_name = malloc(sizeof(uint8_t) * (strlen(argv[i]) + 1));
    memcpy(img_name, argv[i], sizeof(uint8_t) * ((strlen(argv[i]) + 1)));

    npixels = np * nl;
    frame_size = npixels * pixel_size;

    #ifdef _ALMOS_
        nclusters = sysconf(_SC_NCLUSTERS_ONLN);
    #else
        if (nglobal_threads % 4 == 0) {
            nclusters = nglobal_threads / 4;
        }
        else {
            // FIXME: find a stricter condition (which also takes into account the case with 1 thread)
            nclusters = nglobal_threads / 4 + 1;
        }
    #endif
    nlocal_threads = nglobal_threads / nclusters;

    if (nlocal_threads * nclusters != nglobal_threads) {
        fprintf(stderr, "*** Error: the number of local threads must a multiple of the number of clusters.\n");
        exit(1);
    }

    if ((nlocal_threads != 1) && (nlocal_threads != 2) && (nlocal_threads != 4)) {
        fprintf(stderr, "*** Error: the number of local threads must be 1, 2 or 4\n");
        exit(1);
    }

    if (nl % nclusters != 0) {
        fprintf(stderr, "*** Error: the number of clusters must be a divider of the number of lines.\n");
        exit(1);
    }

    if (np % nclusters != 0) {
        fprintf(stderr, "*** Error: the number of clusters must be a divider of the number of columns.\n");
        exit(1);
    }

    printf("\n*** Starting Application EP Filter ***\n");

    /////////////////////////
    // parameters checking //
    /////////////////////////


    // The shared, distributed buffers addresses are computed
    // from the seg_heap_base value defined in the ldscript file
    // and from the cluster increment = 4Gbytes/NB_CLUSTERS.
    // These arrays of pointers are identical and
    // replicated in the stack of each task 

    printf(" - NB_CLUSTERS       = %d\n", nclusters); 
    printf(" - NB_LOCAL_THREADS  = %d\n", nlocal_threads); 
    printf(" - NB_GLOBAL_THREADS = %d\n", nglobal_threads);
    printf(" - NB_PIXELS         = %d\n", npixels);
    printf(" - PIXEL_SIZE        = %d\n", pixel_size);
    printf(" - FRAME_SIZE        = %d\n", frame_size);
    printf("\n");

    A = malloc(sizeof(uint16_t *) * nclusters);
    B = malloc(sizeof(uint32_t *) * nclusters);
    C = malloc(sizeof(uint32_t *) * nclusters);
    D = malloc(sizeof(uint32_t *) * nclusters);
    Z = malloc(sizeof(uint8_t *) * nclusters);

#ifndef PARALLEL_LOAD
    /* Loading image before thread creation because Almos
     * takes 4 000 000 cycles for each page otherwise
     */
    for (i = 0; i < nclusters; i++) {
        A[i] = (uint16_t *) malloc(sizeof(uint16_t) * npixels / nclusters);
        B[i] = (int32_t *)  malloc(sizeof(int32_t) * npixels / nclusters);
        C[i] = (int32_t *)  malloc(sizeof(int32_t) * npixels / nclusters);
        D[i] = (int32_t *)  malloc(sizeof(int32_t) * npixels / nclusters);
        Z[i] = (uint8_t *)  malloc(sizeof(uint8_t) * npixels / nclusters);

        assert(A[i] != NULL);
        assert(B[i] != NULL);
        assert(C[i] != NULL);
        assert(D[i] != NULL);
        assert(Z[i] != NULL);
    }

    //////////////////////////////////////////////////////////////
    // load from disk to A[c] buffers
    //////////////////////////////////////////////////////////////

    printf("\n*** Starting load ***\n");
    FILE * stream = fopen(img_name, "r");
    if (stream == NULL) {
        fprintf(stderr, "*** Error opening file %s\n", img_name);
        exit(1);
    }

    for (i = 0; i < nclusters; i++) {
        fseek(stream, i * frame_size / nclusters, SEEK_SET);
        fread(A[i], pixel_size, npixels / nclusters, stream);
    }
    printf("*** Completed load ***\n");
#endif

    // Barriers initialization
    BARINIT(common_barrier, nglobal_threads);

    CLOCK_APP_CREATE;
    CREATE(process, nglobal_threads);
    WAIT_FOR_END(nglobal_threads);
    CLOCK_APP_JOIN;

    printf("*** Display ***\n");

    FILE * fb;
    int32_t l;
    int32_t tid;
    int32_t first;
    int32_t last;
    #ifdef _ALMOS_
        fb = fopen("/dev/fb0", "w");
        if (fb != NULL) {
            for (tid = 0; tid < nglobal_threads; tid++) {
                first = (tid % 4) * (nl / nglobal_threads);
                last  = first + (nl / nglobal_threads);
                for (l = first; l < last; l++) {
                    fwrite(&TZ((tid / 4), l, 0), 1, np, fb);
                }
            }
            fclose(fb);
        }
        else {
            printf("warning: fb = NULL\n");
        }
    #elif 0
        int32_t p;
        fb = fopen("out_img.ppm", "w");
        fprintf(fb, "P6\n%d %d\n255\n", np, nl);

        for (tid = 0; tid < nglobal_threads; tid++) {
            first = (tid % 4) * (nl / nglobal_threads);
            last  = first + (nl / nglobal_threads);
            for (l = first; l < last; l++) {
                for (p = 0; p < np; p++) {
                    uint8_t grey = (uint8_t) TZ((tid / 4), l, p);
                    fprintf(fb, "%c%c%c", grey, grey, grey);
                }
            }
        }
        fclose(fb);
    #endif

    CLOCK_APP_END;
    CLOCK_FINALIZE(nglobal_threads);
    printf("*** End of EP filter ***\n");
    printf("[NPROCS]   : %16d\n", nglobal_threads);
    PRINT_CLOCK;

    MAIN_END;

    return 0;
}



void process(void * i) {

    const long int tid = (long int) i;                           // processor id == task id
    const uint32_t local_id      = tid % nlocal_threads;         // local task id
    const uint32_t cluster_id    = tid / nlocal_threads;         // cluster task id
    const uint32_t nglobal_threads = nclusters * nlocal_threads; // number of tasks

    const uint32_t lines_per_task     = nl / nglobal_threads; // number of lines per task
    const uint32_t lines_per_cluster  = nl / nclusters;       // number of lines per cluster
    const uint32_t pixels_per_task    = np / nglobal_threads; // number of columns per task
    const uint32_t pixels_per_cluster = np / nclusters;       // number of columns per cluster

    int32_t l; // line index for loops
    int32_t p; // pixel index for loops
    int32_t x; // filter index for loops

    int32_t first, last;

    //int hrange = 10;
    int32_t hrange = 100; // Commenté quand on veut réduire la taille
    int32_t hnorm  = 201;
    
    //////////////////////////////////
    // convolution kernel parameters
    // The content of this section is
    // Philips proprietary information.
    ///////////////////////////////////

    static const int32_t vnorm  = 115;
    static const int32_t vf[35] = { 1, 1, 2, 2, 2,
                   2, 3, 3, 3, 4,
                   4, 4, 4, 5, 5,
                   5, 5, 5, 5, 5,
                   5, 5, 4, 4, 4,
                   4, 3, 3, 3, 2,
                   2, 2, 2, 1, 1 };

    CLOCK_THREAD_START(tid);

#ifdef PARALLEL_LOAD
    if (local_id == 0) {
        // Each proc 0 allocates the memory in its cluster
        A[cluster_id] = (uint16_t *) malloc(sizeof(uint16_t) * npixels / nclusters);
        B[cluster_id] = (int32_t *)  malloc(sizeof(int32_t) * npixels / nclusters);
        C[cluster_id] = (int32_t *)  malloc(sizeof(int32_t) * npixels / nclusters);
        D[cluster_id] = (int32_t *)  malloc(sizeof(int32_t) * npixels / nclusters);
        Z[cluster_id] = (uint8_t *)  malloc(sizeof(uint8_t) * npixels / nclusters);
    }


    BARRIER(common_barrier); // can be removed?

    assert(A[cluster_id] != NULL);
    assert(B[cluster_id] != NULL);
    assert(C[cluster_id] != NULL);
    assert(D[cluster_id] != NULL);
    assert(Z[cluster_id] != NULL);

    //////////////////////////////////////////////////////////////
    // pseudo parallel load from disk to A[c] buffers
    // only task running on processor with (local_id == 0) does it
    //////////////////////////////////////////////////////////////

    if (local_id == 0) {
        PRINTF("\n*** Starting load ***\n");
        FILE * stream = fopen(img_name, "r");
        if (stream == NULL) {
            fprintf(stderr, "*** Error opening file %s\n", img_name);
            exit(1);
        }

        fseek(stream, cluster_id * frame_size / nclusters, SEEK_SET);
        fread(A[cluster_id], pixel_size, npixels / nclusters, stream);

        PRINTF("*** Completed load ***\n");
    }
    BARRIER(common_barrier);
#endif


    //////////////////////////////////////////////////////////
    // parallel horizontal filter : 
    // B <= transpose(FH(A))
    // D <= A - FH(A)
    // Each task computes (nl/nglobal_threads) lines 
    // The image must be extended :
    // if (z<0)    TA(cluster_id,l,z) == TA(cluster_id,l,0)
    // if (z>np-1) TA(cluster_id,l,z) == TA(cluster_id,l,np-1)
    //////////////////////////////////////////////////////////


    PRINTF("\n*** Starting horizontal filter ***\n");

    // this barrier is optional, and will change times measured...
    // if present, it will make global time longer and thread compute time shorter than if it is not present
    BARRIER(common_barrier);

    CLOCK_THREAD_COMPUTE_START(tid);

    // l = absolute line index / p = absolute pixel index  
    // first & last define which lines are handled by a given task(cluster_id,local_id)

    first = (cluster_id * nlocal_threads + local_id) * lines_per_task;
    last  = first + lines_per_task;

    for (l = first; l < last; l++) {
        // src_c and src_l are the cluster index and the line index for A & D
        int src_c = l / lines_per_cluster;
        int src_l = l % lines_per_cluster;

        // We use the specific values of the horizontal ep-filter for optimisation:
        // sum(p) = sum(p-1) + TA[p+hrange] - TA[p-hrange-1]
        // To minimize the number of tests, the loop on pixels is split in three domains 

        int sum_p = (hrange + 2) * TA(src_c, src_l, 0);
        for (x = 1; x < hrange; x++) {
            sum_p = sum_p + TA(src_c, src_l, x);
        }

        // first domain : from 0 to hrange
        for (p = 0; p < hrange + 1; p++) {
            // dst_c and dst_p are the cluster index and the pixel index for B
            int dst_c = p / pixels_per_cluster;
            int dst_p = p % pixels_per_cluster;
            sum_p = sum_p + (int) TA(src_c, src_l, p + hrange) - (int) TA(src_c, src_l, 0);
            TB(dst_c, dst_p, l) = sum_p / hnorm;
            TD(src_c, src_l, p) = (int) TA(src_c, src_l, p) - sum_p / hnorm;
        }
        // second domain : from (hrange+1) to (np-hrange-1)
        for (p = hrange + 1; p < np - hrange; p++) {
            // dst_c and dst_p are the cluster index and the pixel index for B
            int dst_c = p / pixels_per_cluster;
            int dst_p = p % pixels_per_cluster;
            sum_p = sum_p + (int) TA(src_c, src_l, p + hrange) - (int) TA(src_c, src_l, p - hrange - 1);
            TB(dst_c, dst_p, l) = sum_p / hnorm;
            TD(src_c, src_l, p) = (int) TA(src_c, src_l, p) - sum_p / hnorm;
        }
        // third domain : from (np-hrange) to (np-1)
        for (p = np - hrange; p < np; p++) {
            // dst_c and dst_p are the cluster index and the pixel index for B
            int dst_c = p / pixels_per_cluster;
            int dst_p = p % pixels_per_cluster;
            sum_p = sum_p + (int) TA(src_c, src_l, np - 1) - (int) TA(src_c, src_l, p - hrange - 1);
            TB(dst_c, dst_p, l) = sum_p / hnorm;
            TD(src_c, src_l, p) = (int) TA(src_c, src_l, p) - sum_p / hnorm;
        }
    }

    PRINTF("*** Completing horizontal filter ***\n");

    BARRIER(common_barrier);


    //////////////////////////////////////////////////////////
    // parallel vertical filter : 
    // C <= transpose(FV(B))
    // Each task computes (np/nglobal_threads) columns
    // The image must be extended :
    // if (l<0)    TB(cluster_id,p,x) == TB(cluster_id,p,0)
    // if (l>nl-1)   TB(cluster_id,p,x) == TB(cluster_id,p,nl-1)
    //////////////////////////////////////////////////////////

    PRINTF("\n*** Starting vertical filter ***\n");

    // l = absolute line index / p = absolute pixel index
    // first & last define which pixels are handled by a given task(cluster_id,local_id)

    first = (cluster_id * nlocal_threads + local_id) * pixels_per_task;
    last  = first + pixels_per_task;

    for (p = first; p < last; p++) {
        // src_c and src_p are the cluster index and the pixel index for B
        int src_c = p / pixels_per_cluster;
        int src_p = p % pixels_per_cluster;

        int sum_l;

        // We use the specific values of the vertical ep-filter
        // To minimize the number of tests, the nl lines are split in three domains 

        // first domain : explicit computation for the first 18 values
        for (l = 0; l < 18; l++) {
            // dst_c and dst_l are the cluster index and the line index for C
            int dst_c = l / lines_per_cluster;
            int dst_l = l % lines_per_cluster;

            for (x = 0, sum_l = 0; x < 35; x++){
                sum_l = sum_l + vf[x] * TB(src_c, src_p, max(l - 17 + x, 0));
            }
            TC(dst_c, dst_l, p) = sum_l / vnorm;
        }
        // second domain
        for (l = 18; l < nl - 17; l++) {
            // dst_c and dst_l are the cluster index and the line index for C
            int dst_c = l / lines_per_cluster;
            int dst_l = l % lines_per_cluster;

            sum_l = sum_l + TB(src_c, src_p, l + 4)
                + TB(src_c, src_p, l + 8)
                + TB(src_c, src_p, l + 11)
                + TB(src_c, src_p, l + 15)
                + TB(src_c, src_p, l + 17)
                - TB(src_c, src_p, l - 5)
                - TB(src_c, src_p, l - 9)
                - TB(src_c, src_p, l - 12)
                - TB(src_c, src_p, l - 16)
                - TB(src_c, src_p, l - 18);
            TC(dst_c, dst_l, p) = sum_l / vnorm;
        }
        // third domain
        for (l = nl - 17; l < nl; l++) {
            // dst_c and dst_l are the cluster index and the line index for C
            int dst_c = l / lines_per_cluster;
            int dst_l = l % lines_per_cluster;

            sum_l = sum_l + TB(src_c, src_p, min(l + 4, nl - 1))
                + TB(src_c, src_p, min(l + 8, nl - 1))
                + TB(src_c, src_p, min(l + 11, nl - 1))
                + TB(src_c, src_p, min(l + 15, nl - 1))
                + TB(src_c, src_p, min(l + 17, nl - 1))
                - TB(src_c, src_p, l - 5)
                - TB(src_c, src_p, l - 9)
                - TB(src_c, src_p, l - 12)
                - TB(src_c, src_p, l - 16)
                - TB(src_c, src_p, l - 18);
            TC(dst_c, dst_l, p) = sum_l / vnorm;
        }
    }

    PRINTF("*** Completing vertical filter ***\n");


    ////////////////////////////////////////////////////////////////
    // final computation 
    // Z <= D + C
    // Each processor computes (nl/nglobal_threads) lines. 
    ////////////////////////////////////////////////////////////////

    BARRIER(common_barrier);

    PRINTF("\n*** Final computation ***\n");

    first = local_id * lines_per_task;
    last  = first + lines_per_task;

    for (l = first; l < last; l++) {
        for (p = 0; p < np; p++) {
            TZ(cluster_id, l, p) = (uint8_t) (((TD(cluster_id, l, p) + TC(cluster_id, l, p)) >> 8) & 0xFF);
        }
    }

    CLOCK_THREAD_COMPUTE_END(tid);
    CLOCK_THREAD_END(tid);

    PRINTF("*** End of processing ***\n");

} // end process()

// Local Variables:
// tab-width: 4
// c-basic-offset: 4
// c-file-offsets:((innamespace . 0)(inline-open . 0))
// indent-tabs-mode: nil
// End:

// vim: filetype=cpp:expandtab:shiftwidth=4:tabstop=4:softtabstop=4

