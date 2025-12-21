
# Load R data -------------------------------------------------------------

# Set the folder containing your .rds files
cwd <- "C:/Users/cvanb/Desktop/visual_imagery_pupil/study_1/"
input_folder <- paste(cwd, "Rdata", sep="")   # change this to your folder
output_folder <- paste(cwd, "csvdata", sep="")  # optional, can be same as input

# Create output folder if it doesn't exist
if(!dir.exists(output_folder)) {
  dir.create(output_folder)
}

# List all .rds files in the folder
rds_files <- list.files(path = input_folder, pattern = "\\.rds$", full.names = TRUE)

# Loop through each .rds file
for(file in rds_files) {
  # Read the .rds file
  data <- readRDS(file)
  
  # Determine output CSV path
  file_name <- tools::file_path_sans_ext(basename(file))  # removes .rds extension
  output_file <- file.path(output_folder, paste0(file_name, ".csv"))
  
  # Save as CSV
  write.csv(data, output_file, row.names = FALSE, fileEncoding = "UTF-8")
  
  cat("Converted:", file, "->", output_file, "\n")
}

cat("All files converted!\n")

