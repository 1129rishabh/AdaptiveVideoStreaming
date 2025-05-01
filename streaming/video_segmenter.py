# import numpy as np
import cv2
import os


class VideoSegmenter:
    def __init__(self, config):
        self.config = config
        self.segment_duration = config['video']['segment_duration']
        self.available_bitrates = config['video']['available_bitrates']

    def segment_video(self, video_path, output_dir):
        """
        Segment a video file into chunks of specified duration at different bitrates

        Args:
            video_path: path to the input video file
            output_dir: directory to save the segmented files

        Returns:
            dict mapping segment indices to dict of bitrate->filepath
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate frames per segment
        frames_per_segment = int(fps * self.segment_duration)

        # Calculate number of segments
        num_segments = total_frames // frames_per_segment

        # Dictionary to store segment information
        segments = {}

        # Process each segment
        for segment_idx in range(num_segments):
            segment_info = {}

            # Process each bitrate
            for bitrate in self.available_bitrates:
                # Calculate target resolution based on bitrate
                target_height = bitrate
                target_width = int(width * (target_height / height))

                # Output filename
                output_filename = f"segment_{segment_idx}_bitrate_{bitrate}.mp4"
                output_path = os.path.join(output_dir, output_filename)

                # Create VideoWriter object
                # noinspection PyUnresolvedReferences
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

                # Set position in video
                start_frame = segment_idx * frames_per_segment
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                # Read and write frames for this segment
                for _ in range(frames_per_segment):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Resize frame to target resolution
                    resized_frame = cv2.resize(frame, (target_width, target_height))

                    # Write frame to output
                    out.write(resized_frame)

                # Release the writer
                out.release()

                # Store segment info
                segment_info[bitrate] = output_path

            segments[segment_idx] = segment_info

        # Release the video capture
        cap.release()

        return segments

    def get_segment_size(self, segment_path):
        """
        Get the size of a segment file in bytes

        Args:
            segment_path: path to the segment file

        Returns:
            size in bytes
        """
        return os.path.getsize(segment_path)

    def get_segment_bitrate(self, segment_path):
        """
        Estimate the actual bitrate of a segment

        Args:
            segment_path: path to the segment file

        Returns:
            bitrate in kbps
        """
        # Get file size in bytes
        size_bytes = self.get_segment_size(segment_path)

        # Convert to bits
        size_bits = size_bytes * 8

        # Calculate bitrate (bits per second)
        bitrate_bps = size_bits / self.segment_duration

        # Convert to kbps
        bitrate_kbps = bitrate_bps / 1000

        return bitrate_kbps

    def generate_manifest(self, segments, output_path):
        """
        Generate a manifest file for the segmented video (simplified MPD-like format)

        Args:
            segments: dict mapping segment indices to dict of bitrate->filepath
            output_path: path to save the manifest file

        Returns:
            path to the manifest file
        """
        with open(output_path, 'w') as f:
            f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            f.write("<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" type=\"static\">\n")
            f.write(f"  <Period duration=\"{len(segments) * self.segment_duration}s\">\n")

            # Write adaptation set
            f.write("    <AdaptationSet mimeType=\"video/mp4\">\n")

            # Write representations (one per bitrate)
            for bitrate in self.available_bitrates:
                f.write(
                    f"      <Representation id=\"{bitrate}\" width=\"{bitrate * 16 / 9}\" height=\"{bitrate}\" bandwidth=\"{bitrate * 1000}\">\n")
                f.write("        <SegmentList>\n")

                # Write segments
                for segment_idx in sorted(segments.keys()):
                    if bitrate in segments[segment_idx]:
                        segment_path = segments[segment_idx][bitrate]
                        segment_file = os.path.basename(segment_path)
                        segment_size = self.get_segment_size(segment_path)
                        f.write(
                            f"          <SegmentURL media=\"{segment_file}\" mediaRange=\"0-{segment_size - 1}\" />\n")

                f.write("        </SegmentList>\n")
                f.write("      </Representation>\n")

            f.write("    </AdaptationSet>\n")
            f.write("  </Period>\n")
            f.write("</MPD>\n")

        return output_path
