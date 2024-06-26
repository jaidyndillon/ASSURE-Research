#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:51:28 2024

@author: jaidyndillon

This code is to put to use general Python functions and knowledge to extract SNP ID's from the geno.bim file and write them into a new file
"""

with open("/Users/jaidyndillon/Downloads/geno.bim", "r") as geno_data:
    lines = []
    #SNP ID's for each chromosome are respectively placed into their own lists 
    chromosome1 = [] 
    chromosome2 = []
    chromosome3 = []
    chromosome4 = []
    chromosome5 = []
    snps = []
    #IF you want to write the SNP ID's into their own files respective of their chromosome
    #snp1 = [] 
    #snp2 = [] 
    #snp3 = [] 
    #snp4 = [] 
    #snp5 = [] 
    for line in geno_data: 
        line = line.strip()
        lines.append(line)
        if line.startswith("1"): 
            chromosome1.append(line)
        elif line.startswith("2"): 
            chromosome2.append(line)
        elif line.startswith("3"): 
            chromosome3.append(line)
        elif line.startswith("4"): 
            chromosome4.append(line)
        elif line.startswith("5"): 
            chromosome5.append(line)
    for line in chromosome1: 
        #geno.bim has several columns separated by tabs
        #the first two columns are the chromosome and SNP ID (what we want)
        #But since the values increase, you cannot slice at the same place to extract the SNP ID 
        #Or you'd end up either cutting off numbers from the SNP ID or getting extra numbers from other columns
        #These if, else statements ensure proper slicing depending on where the column ends
        if "\t" in line[5:8]: 
            snpid = line[1:7]
            snps.append(snpid.strip())
            #if putting each chromosome's SNP ID into a new file
            #snp1.append(snpid.strip())
        else:
            snpid = line[1:9]
            snps.append(snpid.strip())
            #if putting each chromosome's SNP ID into a new file
            #snp1.append(snpid.strip())
    for line in chromosome2: 
        if "\t" in line[5:8]: 
            snpid2 = line[1:7]
            snps.append(snpid2.strip())
            #if putting each chromosome's SNP ID into a new file
            #snp2.append(snpid.strip())
        else:
            snpid2 = line[1:9]
            snps.append(snpid2.strip())
            #if putting each chromosome's SNP ID into a new file
            #snp2.append(snpid.strip())
    for line in chromosome3: 
        if "\t" in line[5:8]: 
            snpid3 = line[1:7]
            snps.append(snpid3.strip())
            #if putting each chromosome's SNP ID into a new file
            #snp3.append(snpid.strip())
        else:
            snpid3 = line[1:9]
            snps.append(snpid3.strip())
            #if putting each chromosome's SNP ID into a new file
            #snp3.append(snpid.strip())
    for line in chromosome4: 
        if "\t" in line[5:8]: 
            snpid4 = line[1:7]
            snps.append(snpid4.strip())
            #if putting each chromosome's SNP ID into a new file
            #snp4.append(snpid.strip())
        else:
            snpid4 = line[1:9]
            snps.append(snpid4.strip())
            #if putting each chromosome's SNP ID into a new file
            #snp4.append(snpid.strip())
    for line in chromosome5: 
        if "\t" in line[5:8]: 
            snpid5 = line[1:7]
            snps.append(snpid5.strip())
            #if putting each chromosome's SNP ID into a new file
            #snp5.append(snpid.strip())
        else:
            snpid5 = line[1:10]
            snps.append(snpid5.strip())
            #if putting each chromosome's SNP ID into a new file
            #snp5.append(snpid.strip())
with open("/Users/jaidyndillon/Downloads/all_snpids", "w") as all_snps: 
    all_snps.write("%s\n" % snps)
#If creating separate, respective, files 
#with open("/Users/jaidyndillon/Downloads/snp_chrom1", "w") as snp_chrom1: 
    #snp_chrom1.write("%s\n" % snp1)
#with open("/Users/jaidyndillon/Downloads/snp_chrom2", "w") as snp_chrom2: 
    #snp_chrom2.write("%s\n" % snp2)
#with open("/Users/jaidyndillon/Downloads/snp_chrom3", "w") as snp_chrom3: 
    #snp_chrom3.write("%s\n" % snp3)
#with open("/Users/jaidyndillon/Downloads/snp_chrom4", "w") as snp_chrom4: 
    #snp_chrom4.write("%s\n" % snp4)
#with open("/Users/jaidyndillon/Downloads/snp_chrom5", "w") as snp_chrom5: 
    #snp_chrom5.write("%s\n" % snp5)


#You can add a print statement at the end stating that it was completed as well