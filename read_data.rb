require 'csv'

def readData(path, savePath)
  # csv writer
  data = CSV.open(savePath, "w")
  # read data from text file and export to csv
  File.readlines(path).each.with_index do | line, idx |
    # split row by space
    row = line.strip.split
    dim = row.size
    # puts idx.to_s + " " + row[0]

    if idx === 0
      data << row
    else
      # puts row[0]
      data << row
    end
  end

  data.close()
end


if __FILE__ === $0
  filename = "./data/input_fl_12477"
  saveDataPath = "./data/input_fl_12477.csv"

  if !saveDataPath.end_with? (".csv")
    puts "save dir must be csv file(.csv)"
  elsif !File.exist? (filename)
    puts "data file not exists!"
  end
  
  readData(filename, saveDataPath)
end