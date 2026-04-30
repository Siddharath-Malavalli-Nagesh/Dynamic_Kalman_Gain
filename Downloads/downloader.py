#!/usr/bin/env python
#official nclt dataset downloader updated for Python3
#we will be using the data from 2012-01-08
#./downloader.py --date="2012-01-08" --all

#!/usr/bin/env python3

import sys
import os
import subprocess
import argparse
import urllib.request
import re
import ssl

base_url = 'http://robots.engin.umich.edu/nclt/'

dates =[
    '2012-01-08', '2012-01-15', '2012-01-22', '2012-02-02', '2012-02-04',
    '2012-02-05', '2012-02-12', '2012-02-18', '2012-02-19', '2012-03-17',
    '2012-03-25', '2012-03-31', '2012-04-29', '2012-05-11', '2012-05-26',
    '2012-06-15', '2012-08-04', '2012-08-20', '2012-09-28', '2012-10-28',
    '2012-11-04', '2012-11-16', '2012-11-17', '2012-12-01', '2013-01-10',
    '2013-02-23', '2013-04-05'
]

def ensure_output_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def fetch_live_links():
    print("Scraping the live NCLT website for updated URLs...")
    try:
        # Ignore SSL errors just in case the university server cert is expired
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(base_url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, context=ctx).read().decode('utf-8')
        
        # Extract all href links from the website
        links = re.findall(r'href=["\'](.*?)["\']', html)
        
        full_links =[]
        for l in links:
            if l.startswith('http'):
                full_links.append(l)
            else:
                full_links.append(base_url.rstrip('/') + '/' + l.lstrip('/'))
        return full_links
    except Exception as e:
        print(f"Failed to fetch website links: {e}")
        return[]

def get_link(links, date, keywords):
    """Finds a specific link matching both the date and the required file keywords."""
    for l in links:
        if date in l and all(k in l for k in keywords):
            return l
    return None

def download_file(url, out_dir):
    ensure_output_dir(out_dir)
    cmd = ['wget', '--continue', url, '-P', out_dir]
    print(f"\nCalling: {' '.join(cmd)}")
    subprocess.call(cmd)

def main():
    getopt = argparse.ArgumentParser(description='Download NCLT dataset (Dynamic Link Updater)')
    getopt.add_argument('--all', action='store_true', help='Download all data types')
    getopt.add_argument('--lb3', action='store_true', help='Download Ladybug3 Images')
    getopt.add_argument('--sen', action='store_true', help='Download sensor data')
    getopt.add_argument('--vel', action='store_true', help='Download velodyne data')
    getopt.add_argument('--hokuyo', action='store_true', help='Download hokuyo data')
    getopt.add_argument('--gt', action='store_true', help='Download ground truth')
    getopt.add_argument('--gt_cov', action='store_true', help='Download ground truth covariance')
    getopt.add_argument('--date', help='Download specific date')
    
    args = getopt.parse_args()

    if not any([args.all, args.lb3, args.sen, args.vel, args.hokuyo, args.gt, args.gt_cov]):
        print("No data type specified. Use --help to see options.")
        return 1

    # Grab current working links from the site
    live_links = fetch_live_links()
    if not live_links:
        print("Could not fetch links. The university server might be completely down.")
        return 1

    for date in dates:
        if args.date is not None and args.date != date:
            continue
        
        if args.lb3 or args.all:
            url = get_link(live_links, date,['lb3', '.tar.gz'])
            if url: download_file(url, 'images')
            else: print(f"Link not found on NCLT site for {date} images")
            
        if args.sen or args.all:
            url = get_link(live_links, date,['sen', '.tar.gz'])
            if url: download_file(url, 'sensor_data')
            else: print(f"Link not found on NCLT site for {date} sensor data")
            
        if args.vel or args.all:
            url = get_link(live_links, date,['vel', '.tar.gz'])
            if url: download_file(url, 'velodyne_data')
            else: print(f"Link not found on NCLT site for {date} velodyne data")
            
        if args.hokuyo or args.all:
            url = get_link(live_links, date, ['hokuyo', '.tar.gz'])
            if url: download_file(url, 'hokuyo_data')
            else: print(f"Link not found on NCLT site for {date} hokuyo data")
            
        if args.gt or args.all:
            url = get_link(live_links, date, ['groundtruth', '.csv'])
            if url: download_file(url, 'ground_truth')
            else: print(f"Link not found on NCLT site for {date} ground truth")
            
        if args.gt_cov or args.all:
            url = get_link(live_links, date, ['cov', '.csv'])
            if url: download_file(url, 'ground_truth_cov')
            else: print(f"Link not found on NCLT site for {date} ground truth covariance")

    return 0

if __name__ == '__main__':
    sys.exit(main())

