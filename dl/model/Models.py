
from sqlalchemy import Column, String, Float, Integer, ForeignKey, Boolean, PrimaryKeyConstraint, BigInteger
from sqlalchemy.orm import relationship, backref

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import ClauseElement

import numpy
from psycopg2.extensions import register_adapter, AsIs

print('Database initialization')
#engine = create_engine('postgresql://postgres:beadsProject@localhost:5432/dbbeads')
engine = create_engine('postgresql://postgres:beadsProject@localhost:5432/shaking_beads', pool_size=20, max_overflow=0)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Dataset(Base):
    __tablename__ = 'datasets'

    id = Column(Integer, primary_key=True)

    dataset_name = Column(String(50))  # Name of the dataset
    dataset_folder = Column(String(1000), nullable=False, unique=True)  # Folder of the dataset
    dataset_type = Column(String(50))  # Type of the dataset


class Experiment(Base):
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)

    exp_nbeads = Column(Integer, nullable=False)
    exp_velocity = Column(Integer, nullable=False)
    exp_iteration = Column(Integer, nullable=False)
    exp_type = Column(Integer, nullable=False)

    exp_valid = Column(Boolean)
    exp_warning = Column(Boolean)

    exp_folder = Column(String(1000))
    exp_sanity_checked = Column(Boolean)
    exp_ascension_time = Column(Float)
    exp_boiling_time = Column(Float)

    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    dataset = relationship("Dataset", backref=backref('experiments',
                                                     order_by=id))  # https://hackersandslackers.com/sqlalchemy-data-models/

class Frame(Base):
    __tablename__ = 'frames'

    id = Column(Integer, primary_key=True)

    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    experiment = relationship("Experiment", backref=backref("frames", order_by=id))

    frame_img_path = Column(String(1000))  # Name of the dataset
    frame_speed_dish = Column(Float)
    frame_time = Column(Float)
    frame_nbtop = Column(Integer)

    frame_nid = Column(BigInteger, unique=True)

    frame_number = Column(Integer)



class Estimate(Base):
    __tablename__ = 'estimates'

    id = Column(Integer, primary_key=True)

    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    experiment = relationship("Experiment", backref=backref("estimates", order_by=id))

    frame_id = Column(Integer, ForeignKey('frames.id'))
    frame = relationship("Frame", backref=backref("estimates", order_by=id))

    estimate_name = Column(String(50))

    estimate_value_float = Column(Float)
    estimate_value_int = Column(Integer)


class NucEvent(Base):
    __tablename__ = 'nucevents'

    id = Column(Integer, primary_key=True)

    frame_id = Column(Integer, ForeignKey('frames.id'))
    frame = relationship("Frame", backref=backref("nucevents", order_by=id))

    nucevent_movie_path = Column(String)
    nucevent_location = Column(String)


class Bead(Base):
    __tablename__ = 'beads'

    id = Column(Integer, primary_key=True)

    frame_id = Column(Integer, ForeignKey('frames.id'))
    frame = relationship("Frame", backref=backref("beads", order_by=id))

    bead_top = Column(Boolean)

    bead_x = Column(Integer)
    bead_y = Column(Integer)
    bead_r = Column(Integer)
    bead_speed = Column(Float)

    nucevent_id = Column(Integer, ForeignKey('nucevents.id'))
    nucevent = relationship("NucEvent", backref=backref("beads", order_by=id))

    nb_neighbors = Column(Integer)


Base.metadata.create_all(engine)
print("Schema created or initialized")


def adapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)


def adapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)


def adapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)


register_adapter(numpy.float64, adapt_numpy_float64)
register_adapter(numpy.int64, adapt_numpy_int64)
register_adapter(numpy.int32, adapt_numpy_int32)


def get_or_create(session, model, defaults=None, **kwargs):
    instance = session.query(model).filter_by(**kwargs).one_or_none()
    if instance:
        return instance, False
    else:
        params = {k: v for k, v in kwargs.items() if not isinstance(v, ClauseElement)}
        params.update(defaults or {})
        instance = model(**params)
        try:
            session.add(instance)
            session.commit()
        except Exception:  # The actual exception depends on the specific database so we catch all exceptions. This is similar to the official documentation: https://docs.sqlalchemy.org/en/latest/orm/session_transaction.html
            session.rollback()
            instance = session.query(model).filter_by(**kwargs).one()
            return instance, False
        else:
            return instance, True
