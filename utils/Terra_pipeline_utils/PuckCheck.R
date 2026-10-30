library(glue) ; g=glue ; len=length
library(magrittr)
library(purrr)
library(tidyr)
library(dplyr)
library(data.table)
library(stringr)
#library(rlist)
library(ggplot2)
library(cowplot)
library(gridExtra)
library(ShortRead) # sread, quality, id #BiocManager::install("ShortRead", update = FALSE, ask = FALSE)
library(qs)
library(fs)

# path to fastq of SPATIAL LIBRARY after seqeuncing 
    #eg gs://fc-secure-d99fbd65-eb27-4989-95b4-4cf559aa7d36/fastqs/260721_SL-EXB_0892_B23Y3GKLT4/SI-TT-F5_S12_L001_R2_001.fastq.gz
fastq = '/home/gdallagl/myworkdir/XDP/utils/Terra_pipeline_utils/tmp_file_to_puck_check/SI-TT-F5_S12_L001_R2_001.fastq.gz'

# base GCS recon path (constant); only the BCL run name changes per folder
gs_recon_prefix <- "gs://fc-secure-d99fbd65-eb27-4989-95b4-4cf559aa7d36/recon/"

# BCLs to search (each becomes {gs_recon_prefix}{bcl}/, searched recursively for every Puck.csv underneath)
bcl_list <- c(
  "250828_SL-EXG_0494_B232CW3LT4",
  "250904_SL-EXF_0515_A232TCKLT4",
  "250906_SL-EXF_0517_A232H5GLT4",
  "251112_SL-EXR_0160_B23CGTFLT4",
  "251120_SL-EXB_0704_B237FMGLT4",
  "260311_SL-EXD_0812_B23KLY2LT4",
  "260312_SL-EXA_0755_A23KM53LT4",
  "260406_SL-EXE_0817_B23HJ3CLT4",
  "260411_SL-EXF_0671_A23K5MLLT4_lane_34567",
  "260411_SL-EXF_0672_B23K5M5LT4",
  "260512_SL-EXH_0633_A23KYLVLT4",
  "260529_SL-EXG_0701_B23KV7YLT4",
  "260530_SL-EXK_0436_B23KWV5LT4",
  "260610_SL-EXG_0712_A23TTY7LT4"
)
bcl_gs_paths <- g("{gs_recon_prefix}{bcl_list}/")

# local folder where downloaded Puck.csv files are saved
    # each gets renamed from its cloud subpath (bcl dir + sample dir + recon-params dir), since in the cloud they're all just called "Puck.csv"
local_puck_dir <- "/home/gdallagl/myworkdir/XDP/utils/Terra_pipeline_utils/tmp_file_to_puck_check/pucks_from_bcl"
dir_create(local_puck_dir)

gs_puck_paths <- map(bcl_gs_paths, ~ system(g("gsutil ls -r {.x}"), intern = TRUE)) %>%
  unlist(use.names = FALSE) %>%
  keep(~ str_detect(., "Puck\\.csv$"))
stopifnot(len(gs_puck_paths) > 0)

local_names <- gs_puck_paths %>% str_remove(fixed(gs_recon_prefix)) %>% str_replace_all("/", "_")
local_paths <- path(local_puck_dir, local_names)

walk2(gs_puck_paths, local_paths, function(src, dst) {
  if (!file_exists(dst)) system(g("gsutil cp {src} {dst}"))
})

puck_files <- local_paths

R = fastq %>% FastqStreamer(n = 1000000) %>% yield %>% sread %>% subseq(start=1, end=36) %>% as.data.table

# Determine bead structure
V10 <- R[, sum(str_sub(x,9,26)=="TCTTCAGCGTTCCCGAGA")]
V17 <- R[, sum(str_sub(x,10,27)=="TCTTCAGCGTTCCCGAGA")]
V15 <- R[, sum(str_sub(x,16,25)=="CTGTTTCCTG")]
V16 <- R[, sum(str_sub(x,18,27)=="CTGTTTCCTG")]

if (V10 == max(V10,V17,V15,V16)) {
    print("V10 recon") # if in-situ, make sb2=6
    R <- separate_wider_position(R, x, c(sb1=8, up=18, sb2=7), too_many="drop")
    print(g("UP matching rate: {sum(R$up == 'TCTTCAGCGTTCCCGAGA')/nrow(R) * 100}%"))
    R %<>% filter(up == 'TCTTCAGCGTTCCCGAGA')
    R %<>% transmute(sb=paste0(sb1, sb2))
} else if (V17 == max(V10,V17,V15,V16)) {
    print("V17")
    R <- separate_wider_position(R, x, c(sb1=9, up=18, sb2=8), too_many="drop")
    print(g("UP matching rate: {sum(R$up == 'TCTTCAGCGTTCCCGAGA')/nrow(R) * 100}%"))
    R %<>% filter(up == 'TCTTCAGCGTTCCCGAGA')
    R %<>% transmute(sb=paste0(sb1, sb2))
} else if (V15 == max(V10,V17,V15,V16)) {
    print("V15")
    R <- separate_wider_position(R, x, c(sb=15, up=10), too_many="drop")
    print(g("UP matching rate: {sum(R$up == 'CTGTTTCCTG')/nrow(R) * 100}%"))
    R %<>% filter(up == 'CTGTTTCCTG')
    R %<>% select(-up)
} else if (V16 == max(V10,V17,V15,V16)) {
    print("V16")
    R <- separate_wider_position(R, x, c(sb=17, up=10), too_many="drop")
    print(g("UP matching rate: {sum(R$up == 'CTGTTTCCTG')/nrow(R) * 100}%"))
    R %<>% filter(up == 'CTGTTTCCTG')
    R %<>% select(-up)
}

pucks <- map(puck_files, ~fread(., select=1))
for (puck in pucks) {setnames(puck, "V1", "sb")}

res <- map_dbl(pucks, ~sum(R$sb %in% .$sb)/nrow(R))


results <- data.frame(puck = puck_files, matching_rate = res) %>% arrange(desc(matching_rate))
print(results, row.names = FALSE)
